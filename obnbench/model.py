"""obnbench/model.py  – drop-in replacement for the original file.

Changes vs. upstream krishnanlab/obnbench:
  • ``build_mp_module`` now returns ``SGCNMPModule`` when
    ``cfg.model.mp_type == 'SGCN'``, bypassing the generic ``MPModule``.
  • Added module-level SGCN constants and ``_sample_subgraph_nodes`` helper
    (ported from SGCN ``src/train.py``).
  • Added ``SGCNModelModule(ModelModule)`` – a Lightning module that implements
    the full SGCN algorithm as a multi-subgraph, multi-model training and
    parameter aggregation framework:

      For each epoch:
        Stage 1 – Local training:
          Sample R subgraphs from the full graph (dynamic, per-epoch).
          For each subgraph:
            - Initialize local model from global parameters θ̃^(t-1).
            - Train locally for L gradient steps → local parameters θ̂_r.
            - Score on a held-out validation mini-batch → s_r.
        Stage 2 – Parameter aggregation:
          Discard the bottom ``truncation_ratio`` fraction of subgraphs (by s_r).
          Aggregate kept local parameters:
            θ̃^(t) = Σ w_r * θ̂_r,  w_r ∝ s_r  (softmax / uniform / linear).

    The four core stages are exposed as named, testable methods:
      sample_subgraphs()      – dynamic subgraph sampling with coverage tracking
      train_local_model()     – local gradient descent on a single subgraph
      evaluate_submodel()     – validation quality score for a local model
      aggregate_parameters()  – truncation + weighted parameter aggregation
"""

import copy
import math
import os.path as osp
import time
from collections import OrderedDict
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import obnb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import subgraph as pyg_subgraph

import obnbench.metrics
from obnbench import optimizers, schedulers
from obnbench.model_layers import feature_encoders, mp_layers
from obnbench.model_layers.post_proc import CorrectAndSmooth, FeaturePropagation

# ─────────────────────────── SGCN module-level constants ─────────────────────

# Fraction of n_sample used as BFS seed nodes (1/20 = 5 %)
_SGCN_SEED_RATIO = 20
# Hard upper bound on BFS hops for random_walk sampling
_SGCN_RANDOM_WALK_MAX_HOPS = 10
# Minimum training nodes injected when a sampled subgraph contains none
_SGCN_MIN_TRAIN_NODES = 32
# Validation nodes sampled for the per-subgraph quality score used in truncation
_SGCN_VAL_SAMPLE_SIZE = 512

# ─────────────────────────── SGCN subgraph sampling ──────────────────────────


def _sample_subgraph_nodes(
    edge_index,
    n_nodes,
    train_idx_cpu,
    method,
    n_sample,
    subgraph_max_nodes=None,
    unsampled_nodes=None,
):
    """Return a 1-D sorted LongTensor of sampled node indices.

    Ported directly from ``SGCN/src/train.py`` with no algorithmic changes.

    Supported methods
    -----------------
    random_node  – uniformly sample *n_sample* nodes at random.
    random_edge  – sample random edges and collect their incident nodes.
    random_walk  – BFS expansion from random training-set seeds (no hop cap).
    snowball     – BFS expansion capped at 2 hops from random seeds.

    Parameters
    ----------
    edge_index : LongTensor [2, E]
        Graph edge index on CPU.
    n_nodes : int
        Total number of nodes.
    train_idx_cpu : LongTensor
        1-D tensor of training node indices (CPU).
    method : str
        Sampling method name (see above).
    n_sample : int
        Target number of nodes.  Overridden by *subgraph_max_nodes* when set.
    subgraph_max_nodes : int or None
        Hard upper bound on returned nodes.  Takes priority over *n_sample*.
    unsampled_nodes : 1-D LongTensor or None
        Nodes not yet covered in the current epoch.  When provided, each
        sampling method prioritises them so that successive subgraphs
        collectively cover the whole graph before revisiting sampled regions.
    """
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    n_sample = min(n_sample, n_nodes)

    has_priority = unsampled_nodes is not None and len(unsampled_nodes) > 0

    if method == "random_node":
        if has_priority:
            n_priority = len(unsampled_nodes)
            if n_priority >= n_sample:
                perm = torch.randperm(n_priority)[:n_sample]
                return unsampled_nodes[perm].sort().values
            remaining = n_sample - n_priority
            sampled_mask = torch.ones(n_nodes, dtype=torch.bool)
            sampled_mask[unsampled_nodes] = False
            sampled_pool = sampled_mask.nonzero(as_tuple=False).squeeze(1)
            perm = torch.randperm(len(sampled_pool))[:remaining]
            return torch.cat([unsampled_nodes, sampled_pool[perm]]).sort().values
        perm = torch.randperm(n_nodes)[:n_sample]
        return perm.sort().values

    elif method == "random_edge":
        n_edges = edge_index.shape[1]
        if has_priority:
            prio_mask = torch.zeros(n_nodes, dtype=torch.bool)
            prio_mask[unsampled_nodes] = True
            edge_has_prio = prio_mask[edge_index[0]] | prio_mask[edge_index[1]]
            prio_edges = edge_has_prio.nonzero(as_tuple=False).squeeze(1)
            other_edges = (~edge_has_prio).nonzero(as_tuple=False).squeeze(1)

            n_prio_sample = min(n_sample * 2, len(prio_edges))
            perm_p = torch.randperm(len(prio_edges))[:n_prio_sample]
            chosen = edge_index[:, prio_edges[perm_p]].flatten().unique()

            if len(chosen) < n_sample and len(other_edges) > 0:
                extra_e = other_edges[
                    torch.randperm(len(other_edges))[: n_sample * 2]
                ]
                extra_n = edge_index[:, extra_e].flatten().unique()
                chosen = torch.cat([chosen, extra_n]).unique()
        else:
            edge_perm = torch.randperm(n_edges)[: min(n_sample * 2, n_edges)]
            chosen = edge_index[:, edge_perm].flatten().unique()

        if len(chosen) < n_sample:
            extra = torch.randperm(n_nodes)[: n_sample - len(chosen)]
            chosen = torch.cat([chosen, extra]).unique()
        return chosen[:n_sample].sort().values

    elif method in ("random_walk", "snowball"):
        if has_priority:
            prio_train = unsampled_nodes[torch.isin(unsampled_nodes, train_idx_cpu)]
            seed_pool = prio_train if len(prio_train) > 0 else unsampled_nodes
        else:
            seed_pool = train_idx_cpu

        n_seeds = min(max(n_sample // _SGCN_SEED_RATIO, 1), len(seed_pool))
        seed_perm = torch.randperm(len(seed_pool))[:n_seeds]
        seeds = seed_pool[seed_perm]

        visited = torch.zeros(n_nodes, dtype=torch.bool)
        visited[seeds] = True

        row, col = edge_index
        max_hops = 2 if method == "snowball" else _SGCN_RANDOM_WALK_MAX_HOPS

        for _ in range(max_hops):
            if int(visited.sum()) >= n_sample:
                break
            mask = visited[row]
            new_nodes = col[mask]
            visited[new_nodes] = True

        visited_nodes = visited.nonzero(as_tuple=False).squeeze(1)

        if len(visited_nodes) > n_sample:
            perm = torch.randperm(len(visited_nodes))[:n_sample]
            return visited_nodes[perm].sort().values
        elif len(visited_nodes) < n_sample:
            unvisited = (~visited).nonzero(as_tuple=False).squeeze(1)
            extra_perm = torch.randperm(len(unvisited))[: n_sample - len(visited_nodes)]
            return torch.cat([visited_nodes, unvisited[extra_perm]]).sort().values

        return visited_nodes.sort().values

    else:
        raise ValueError(
            f"Unknown subsampling_method: {method!r}. "
            "Choose from: 'random_node', 'random_edge', 'random_walk', 'snowball'."
        )


# ─────────────────────────── original model registries ───────────────────────

act_register = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "elu": nn.ELU,
}

norm_register = {
    "BatchNorm": pygnn.BatchNorm,
    "LayerNorm": pygnn.LayerNorm,
    "PairNorm": pygnn.PairNorm,
    "DiffGroupNorm": pygnn.DiffGroupNorm,
    "none": nn.Identity,
}


# ─────────────────────────── original ModelModule ────────────────────────────


class ModelModule(pl.LightningModule):

    splits = ["train", "val", "test"]

    def __init__(self, cfg: DictConfig, node_ids: List[str], task_ids: List[str]):
        super().__init__()

        # Save IDs info for logging
        self.node_ids = node_ids
        self.task_ids = task_ids

        self.save_hyperparameters(cfg)  # register cfg as self.hparams
        self.setup_metrics()

        # Build model
        self.feature_encoder = build_feature_encoder(cfg)
        self.mp_layers = build_mp_module(cfg)
        self.pred_head = build_pred_head(cfg)
        self.post_prop, self.pred_act, self.post_cands = build_post_proc(cfg)

    def setup_metrics(self):
        self.metrics = nn.ModuleDict()
        self.final_pred, self.final_true, self.final_scores = {}, {}, []
        metric_kwargs = {
            "task": "multilabel",
            "num_labels": self.hparams._shared.dim_out,
            "validate_args": False,
        }
        for split in self.splits:
            for metric_name in self.hparams.metric.options:
                metric_cls = getattr(obnbench.metrics, metric_name)

                # Metrics to be calculated step/epoch-wise
                self.metrics[f"{split}/{metric_name}"] = \
                    metric_cls(average="macro", **metric_kwargs)

                # Metrics for final evaluation (record perf of individual task)
                self.metrics[f"final/{split}_{metric_name}"] = \
                    metric_cls(average="none", **metric_kwargs)

            self.final_pred[split], self.final_true[split] = [], []

    def configure_optimizers(self):
        optimizer_cls = getattr(optimizers, self.hparams.optim.optimizer)
        optimizer_kwargs = dict(self.hparams.optim.optimizer_kwargs or {})
        if (weight_decay := self.hparams.optim.weight_decay) is not None:
            optimizer_kwargs["weight_decay"] = weight_decay
        optimizer = optimizer_cls(
            self.parameters(),
            lr=self.hparams.optim.lr,
            **optimizer_kwargs,
        )

        lr_scheduler_config = {"optimizer": optimizer}

        if self.hparams.optim.scheduler != "none":
            scheduler_cls = getattr(schedulers, self.hparams.optim.scheduler)
            scheduler_kwargs = dict(self.hparams.optim.scheduler_kwargs or {})

            eval_interval = self.hparams.trainer.eval_interval
            if (patience := scheduler_kwargs.get("patience", None)):
                # Rescale the scheduler patience for ReduceLROnPlateau to the
                # factor w.r.t. the evaluation interval
                scheduler_kwargs["patience"] = ceil(patience / eval_interval)

            scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

            lr_scheduler_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": f"val/{self.hparams.metric.best}",
                "frequency": eval_interval,
            }

        return lr_scheduler_config

    def forward(self, batch):
        batch = self.feature_encoder(batch)
        batch = self.mp_layers(batch)
        batch = self.pred_head(batch)
        pred, true = self._post_processing(batch)  # FIX: move to end of steps?
        return pred, true

    def _post_processing(self, batch):
        pred, true = batch.x, batch.y

        if self.post_prop is not None:
            pred = self.post_prop(
                pred,
                batch.edge_index,
                batch.edge_weight,
            )

        pred = self.pred_act(pred)

        if not self.training and self.post_cands is not None:
            pred = self.post_cands(
                pred,
                true,
                batch.train_mask.squeeze(-1),
                batch.edge_index,
                batch.edge_weight,
            )

        # Apply split masking
        if batch.split is not None:
            mask = batch[f"{batch.split}_mask"].squeeze(-1)
            pred, true = pred[mask], true[mask]

        return pred, true

    def _shared_step(self, batch, split, final: bool = False):
        tic = time.perf_counter()
        batch.split = split

        # NOTE: split masking is done in _post_processing
        pred, true = self(batch)

        logger_opts = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "batch_size": pred.shape[0],
        }

        # Compute classification loss for training
        if split == "train":
            loss = F.binary_cross_entropy(pred, true)
            self.log(f"{split}/loss", loss.item(), **logger_opts)
            self._maybe_log_grad_norm(logger_opts)
        else:
            loss = None

        self._maybe_log_metrics(pred, true, split, logger_opts)
        self._maybe_prepare_final_metrics(pred, true, split, final)

        self.log(f"{split}/time_epoch", time.perf_counter() - tic, **logger_opts)

        return loss

    def _maybe_log_grad_norm(self, logger_opts):
        if (not self.training) or (not self.hparams.trainer.watch_grad_norm):
            return

        grad_norms = [
            p.grad.detach().norm(2)
            for p in self.parameters() if p.grad is not None
        ]
        if grad_norms:
            grad_norm = torch.stack(grad_norms).norm(2).item()
            self.log("train/grad_norm", grad_norm, **logger_opts)

    @torch.no_grad()
    def _maybe_log_metrics(self, pred, true, split, logger_opts):
        if (
            (self.current_epoch + 1) % self.hparams.trainer.eval_interval != 0
            and split == "train"
        ):
            return

        for metric_name, metric_obj in self.metrics.items():
            if metric_name.startswith(f"{split}/"):
                metric_obj(pred, true)
                self.log(metric_name, metric_obj, **logger_opts)

    def _maybe_prepare_final_metrics(self, pred, true, split, final):
        if final:
            self.final_pred[split].append(pred.detach())
            self.final_true[split].append(true.detach())

    def _compute_final_metrics(self, split):
        pred = torch.vstack(self.final_pred[split])
        true = torch.vstack(self.final_true[split])

        for metric_name, metric_obj in self.metrics.items():
            prefix = f"final/{split}_"
            if not metric_name.startswith(prefix):
                continue

            score_type = metric_name.replace(prefix, "")
            scores = metric_obj(pred, true).tolist()
            for task_id, task_score in zip(self.task_ids, scores):
                self.final_scores.append(
                    {
                        "split": split,
                        "task_id": task_id,
                        "score_type": score_type,
                        "score_value": task_score,
                    }
                )

            metric_obj.reset()

    def log_final_results(self):
        score_df = pd.DataFrame(self.final_scores)

        for logger in self.loggers:
            if isinstance(logger, pl.loggers.CSVLogger):
                out_path = osp.join(logger.log_dir, "final_scores.csv")
                score_df.to_csv(out_path)
                obnb.logger.info(f"Final results saved to {out_path}")

            elif isinstance(logger, pl.loggers.WandbLogger):
                logger.log_table("final_scores", dataframe=score_df)

            else:
                obnb.logger.error(f"Unknown logger type {type(logger)}")

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        self._shared_step(batch, split="val")
        # HACK: Enable early testing that was deliberaly disabled by Lightning
        # https://github.com/Lightning-AI/lightning/issues/5245
        self._shared_step(batch, split="test")

    def on_test_epoch_start(self):
        # Reset final outputs
        for obj in [self.final_pred, self.final_true]:
            for split in obj:
                obj[split].clear()
        self.final_scores.clear()

    def test_step(self, batch, *args, **kwargs):
        for split in self.splits:
            self._shared_step(batch, split=split, final=True)

    def on_test_epoch_end(self, *args, **kwargs):
        for split in self.splits:
            self._compute_final_metrics(split)


# ─────────────────────────── original MPModule ───────────────────────────────


class MPModule(nn.Module):

    _residual_opts = [
        "none",
        "skipsum",
        "skipsumbnorm",
        "skipsumlnorm",
        "catlast",
        "catall",
        "catcompact",
    ]

    def __init__(
        self,
        mp_cls: nn.Module,
        dim: int,
        num_layers: int,
        dropout: float = 0.0,
        norm_type: str = "BatchNorm",
        act: str = "relu",
        act_first: bool = False,
        residual_type: str = "none",
        mp_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type
        self.act = act
        self.act_first = act_first

        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs or {}
        if norm_type == "LayerNorm":
            self.norm_kwargs.setdefault("mode", "node")
        elif norm_type == "DiffGroupNorm":
            self.norm_kwargs.setdefault("groups", 6)
        if norm_type != "PairNorm":
            # Need to pass feature dimension except for PairNorm
            self.norm_kwargs["in_channels"] = None

        # Set residual_type last to make sure we have registered all params
        # Setting residual_type will automatically set up the forward function
        # and the dimensions for the hidden layers.
        self.residual_type = residual_type

        # Set up the message passing layers using the dimensions prepared
        self.layers = nn.ModuleList()
        self.res_norms = nn.ModuleList()
        mp_kwargs = mp_kwargs or {}
        for dim_in in self._layer_dims:
            self._build_layer(mp_cls, dim_in, dim, mp_kwargs)

    def _build_layer(
        self,
        mp_cls: nn.Module,
        dim_in: int,
        dim_out: int,
        mp_kwargs: Dict[str, Any],
    ) -> nn.Module:
        conv_layer = mp_cls(dim_in, dim_out, **mp_kwargs)
        activation = act_register.get(self.act)()
        dropout = nn.Dropout(self.dropout)

        # Check if 'in_channels' is set to determine whether we need to pass
        # the feature dimension to initialize the normalization layer
        if "in_channels" in self.norm_kwargs:
            self.norm_kwargs["in_channels"] = dim_out
        norm_layer = norm_register.get(self.norm_type)(**self.norm_kwargs)

        # Graph convolution layer
        new_layer = nn.ModuleDict()
        new_layer["conv"] = conv_layer

        # Post convolution layers
        post_conv = []
        if self.act_first:
            post_conv.extend([("act", activation), "norm", norm_layer])
        else:
            post_conv.extend([("norm", norm_layer), ("act", activation)])
        post_conv.append(("dropout", dropout))
        new_layer["post_conv"] = nn.Sequential(OrderedDict(post_conv))
        self.layers.append(new_layer)

        # Residual normalizations
        if self.residual_type == "skipsumbnorm":
            self.res_norms.append(norm_register["BatchNorm"](dim_out))
        elif self.residual_type == "skipsumlnorm":
            self.res_norms.append(norm_register["LayerNorm"](dim_out, mode="node"))

    def extra_repr(self) -> str:
        return f"residual_type: {self.residual_type}"

    @property
    def residual_type(self) -> str:
        return self._residual_type

    @residual_type.setter
    def residual_type(self, val: str):
        if val == "none":
            self._forward = self._stack_forward
            self._layer_dims = [self.dim] * self.num_layers
        elif val in ["skipsum", "skipsumbnorm", "skipsumlnorm"]:
            self._forward = self._skipsum_forward
            self._layer_dims = [self.dim] * self.num_layers
        elif val == "catlast":
            self._forward = self._catlast_forward
            self._layer_dims = (
                [self.dim] * (self.num_layers - 1)
                + [self.dim * self.num_layers]
            )
        elif val == "catall":
            self._forward = self._catall_forward
            self._layer_dims = [self.dim * (i + 1) for i in range(self.num_layers)]
        else:
            raise ValueError(
                f"Unknown residual type {val!r}, available options are:\n"
                f"    {self._residual_opts}",
            )
        self._residual_type = val

    @staticmethod
    def _layer_forward(layer, batch):
        batch = layer["conv"](batch)
        batch.x = layer["post_conv"](batch.x)
        return batch

    def _stack_forward(self, batch):
        for layer in self.layers:
            batch = self._layer_forward(layer, batch)
        return batch

    def _skipsum_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = layer["conv"](batch)
            if self.res_norms:
                batch.x = self.res_norms[i](batch.x)
            batch.x = batch.x + x_prev
            batch.x = layer["post_conv"](batch.x)
        return batch

    def _catlast_forward(self, batch):
        xs = []
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                batch.x = torch.cat([batch.x] + xs, dim=1)
            batch = self._layer_forward(layer, batch)
            if i < self.num_layers - 1:
                xs.append(batch.x)
        return batch

    def _catall_forward(self, batch):
        for i, layer in enumerate(self.layers):
            x_prev = batch.x
            batch = self._layer_forward(layer, batch)
            if i < self.num_layers - 1:
                batch.x = torch.cat([batch.x, x_prev], dim=1)
        return batch

    def forward(self, batch):
        return self._forward(batch)


# ─────────────────────────── original PredictionHeadModule ───────────────────


class PredictionHeadModule(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_layers: int = 1,
        dim_inner: int = 128,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = pygnn.MLP(
                in_channels=dim_in,
                out_channels=dim_out,
                hidden_channels=dim_inner,
                num_layers=num_layers,
                act="relu",
                norm="batch_norm",
                bias=True,
                plain_last=True,
            )
        else:
            self.layers = nn.Identity()

    def forward(self, batch):
        batch.x = self.layers(batch.x)
        return batch


# ─────────────────────────── SGCN Lightning module ───────────────────────────


class SGCNModelModule(ModelModule):
    """Lightning module implementing SGCN's multi-subgraph training algorithm.

    SGCN is NOT a standard GCN trained on a fixed graph.  It is a
    multi-subgraph, multi-model training framework with parameter-level
    aggregation (analogous to federated averaging / ensemble distillation).

    Algorithm (per epoch t)
    -----------------------
    Stage 1 – Local training on R independent subgraphs:
      For r = 1 … R:
        1. Sample subgraph G_r from the full graph (dynamic, per epoch).
        2. Initialize a local model from the current global parameters θ̃^(t-1).
        3. Train for L gradient steps on G_r  →  local parameters θ̂_r^(t).
        4. Evaluate local model quality on a validation mini-batch  →  score s_r^(t).

    Stage 2 – Global parameter aggregation:
        5. Discard the bottom ``truncation_ratio`` fraction of subgraphs (by s_r).
        6. Compute weighted average of kept local parameters:
             θ̃^(t) = Σ w_r * θ̂_r^(t),   w_r ∝ s_r^(t)

    The four core operations are exposed as named, independently testable methods:
      ``sample_subgraphs()``     – dynamic per-epoch subgraph sampling
      ``train_local_model()``    – local gradient descent on a single subgraph
      ``evaluate_submodel()``    – validation quality score for a local model
      ``aggregate_parameters()`` – truncation + weighted parameter aggregation

    SGCN hyperparameters are read from ``cfg.model.sgcn.*``.  See
    ``conf/model/sgcn.yaml`` for default values and documentation.

    Notes
    -----
    * Uses ``automatic_optimization = False`` (Lightning manual-optimization
      mode) so the per-subgraph gradient loop and the final state-load are
      fully under our control.
    * Gradient clipping (``cfg.trainer.gradient_clip_val``) is applied
      manually via ``self.clip_gradients`` before each ``optimizer.step()``.
    * The LR scheduler is still stepped automatically by Lightning via the
      ``lr_scheduler_config`` returned from ``configure_optimizers``; the
      monitored metric (e.g. ``val/APOP``) is logged in the inherited
      ``validation_step``.
    * Feature encoders that assume full-graph inputs (``AdjEmbBag``,
      ``Embedding``) are **not** compatible with subgraph training.  Use
      ``OneHotLogDeg``, ``SVD``, ``LapEigMap``, or similar encoders instead.
    """

    def __init__(self, cfg: DictConfig, node_ids: List[str], task_ids: List[str]):
        super().__init__(cfg, node_ids, task_ids)

        # Switch to manual optimization so we control the inner subgraph loop
        self.automatic_optimization = False

        # Parse SGCN-specific hyperparameters (all optional with sensible defaults)
        sgcn_cfg: Dict[str, Any] = dict(cfg.model.get("sgcn") or {})
        self._n_subgraphs: int = int(sgcn_cfg.get("n_subgraphs", 5))
        self._local_epochs: int = int(sgcn_cfg.get("local_epochs", 5))
        self._subsampling_method: str = sgcn_cfg.get("subsampling_method", "random_node")
        self._subgraph_max_nodes: int = int(sgcn_cfg.get("subgraph_max_nodes", 2048))
        self._max_subgraph_edges: int = int(sgcn_cfg.get("max_subgraph_edges", 0))
        self._subgraph_ratio: float = float(sgcn_cfg.get("subgraph_ratio", 0.5))
        self._truncation_ratio: float = float(sgcn_cfg.get("truncation_ratio", 0.2))
        self._aggregation_method: str = sgcn_cfg.get("aggregation_method", "sgcn")

    # ── low-level helpers (subgraph construction + forward) ──────────────────

    def _make_sub_batch(self, batch, node_idx: torch.Tensor) -> Data:
        """Slice *batch* to the nodes in *node_idx* and return a new Data object.

        The returned object contains:
        - Relabeled ``edge_index`` and filtered ``edge_weight`` (via
          :func:`pyg_subgraph`).
        - ``y``, ``train_mask``, ``val_mask`` sliced to ``node_idx``.
        - All ``rawfeat_*`` attributes sliced to ``node_idx`` so the feature
          encoder can run on the subgraph unchanged.

        An optional hard edge cap (``_max_subgraph_edges``) is enforced here
        by randomly dropping excess edges.
        """
        device = batch.edge_index.device
        node_idx_dev = node_idx.to(device)
        n_nodes = batch.num_nodes

        # Extract induced subgraph with relabeled node IDs
        edge_weight = getattr(batch, "edge_weight", None)
        edge_index_sub, edge_weight_sub = pyg_subgraph(
            node_idx_dev,
            batch.edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=n_nodes,
        )

        # Enforce hard edge cap (0 = disabled)
        if self._max_subgraph_edges > 0:
            n_e = edge_index_sub.size(1)
            if n_e > self._max_subgraph_edges:
                perm = torch.randperm(n_e, device=device)[: self._max_subgraph_edges]
                edge_index_sub = edge_index_sub[:, perm]
                if edge_weight_sub is not None:
                    edge_weight_sub = edge_weight_sub[perm]

        sub_data = Data(
            edge_index=edge_index_sub,
            edge_weight=edge_weight_sub,
            y=batch.y[node_idx_dev],
            train_mask=batch.train_mask[node_idx_dev],
            val_mask=batch.val_mask[node_idx_dev],
        )
        sub_data.num_nodes = int(node_idx.shape[0])

        # Copy all precomputed raw-feature attributes so the feature encoder
        # can encode subgraph nodes without modification.
        for key in batch.keys():
            if key.startswith("rawfeat_"):
                sub_data[key] = batch[key][node_idx_dev]

        return sub_data

    def _forward_subgraph(self, sub_batch: Data) -> torch.Tensor:
        """Full model forward on *sub_batch*; returns sigmoid predictions."""
        sub_batch = self.feature_encoder(sub_batch)
        sub_batch = self.mp_layers(sub_batch)
        sub_batch = self.pred_head(sub_batch)
        return self.pred_act(sub_batch.x)  # sigmoid

    # ── Stage 1 helpers: subgraph sampling + local training ───────────────────

    def sample_subgraphs(
        self,
        batch: Data,
        n_subgraphs: int,
        n_sample: int,
        edge_index_cpu: torch.Tensor,
        train_idx_cpu: torch.Tensor,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, Data]]:
        """Sample *n_subgraphs* independent subgraphs from the full graph.

        Implements epoch-level coverage tracking: successive samplers are
        biased toward nodes not yet visited this epoch, so the union of all
        sampled subgraphs covers the full graph before revisiting any region.

        Each candidate subgraph is checked for training nodes; subgraphs that
        contain none are silently skipped.

        Parameters
        ----------
        batch : Data
            Full-graph Data object.
        n_subgraphs : int
            Number of subgraphs to sample.
        n_sample : int
            Target node budget per subgraph.
        edge_index_cpu : LongTensor
            ``batch.edge_index`` pre-moved to CPU (avoids repeated transfers).
        train_idx_cpu : LongTensor
            Sorted training-node indices on CPU.
        device : torch.device
            Device on which the returned ``sub_batch`` tensors live.

        Returns
        -------
        list of (node_idx, sub_batch)
            ``node_idx`` is a 1-D CPU LongTensor of sampled node indices.
            ``sub_batch`` is the corresponding induced-subgraph Data on *device*.
        """
        n_nodes = batch.num_nodes
        # Track which nodes have been visited this epoch so that successive
        # samplers prioritise unvisited regions (full-graph coverage).
        epoch_sampled_mask = torch.zeros(n_nodes, dtype=torch.bool)
        subgraphs: List[Tuple[torch.Tensor, Data]] = []

        for _ in range(n_subgraphs):
            # Bias sampler toward unvisited nodes for better coverage
            unsampled = epoch_sampled_mask.logical_not().nonzero(as_tuple=False).squeeze(1)
            unsampled_nodes = unsampled if unsampled.numel() > 0 else None

            node_idx = _sample_subgraph_nodes(
                edge_index_cpu,
                n_nodes,
                train_idx_cpu,
                self._subsampling_method,
                n_sample,
                subgraph_max_nodes=self._subgraph_max_nodes,
                unsampled_nodes=unsampled_nodes,
            )

            # Guarantee at least a few training nodes are present in the subgraph
            if not torch.isin(node_idx, train_idx_cpu).any():
                extra = train_idx_cpu[
                    torch.randperm(len(train_idx_cpu))[
                        : min(_SGCN_MIN_TRAIN_NODES, len(train_idx_cpu))
                    ]
                ]
                node_idx = torch.cat([node_idx, extra]).unique().sort().values

            epoch_sampled_mask[node_idx] = True

            # Build the induced subgraph and move to target device
            sub_batch = self._make_sub_batch(batch, node_idx).to(device)
            if not sub_batch.train_mask.squeeze(-1).any():
                continue  # skip subgraphs with no training nodes

            subgraphs.append((node_idx, sub_batch))

        return subgraphs

    def train_local_model(
        self,
        sub_batch: Data,
        optimizer,
        epoch_init_state: Dict[str, torch.Tensor],
        clip_val: Optional[float],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Train a fresh local model on a single subgraph for L gradient steps.

        This implements the per-subgraph local training stage of SGCN:
          - Reset the model to the current global parameters θ̃^(t-1).
          - Run ``_local_epochs`` gradient steps on *sub_batch*.
          - Return the resulting local parameters θ̂_r and the final loss.

        Parameters
        ----------
        sub_batch : Data
            Induced subgraph Data object (already on the target device).
        optimizer : torch.optim.Optimizer
            The Lightning-managed optimizer (accessed via ``self.optimizers()``).
        epoch_init_state : dict
            CPU copy of the global parameters θ̃^(t-1) snapshotted at epoch start.
        clip_val : float or None
            Gradient-norm clipping threshold; ``None`` disables clipping.

        Returns
        -------
        local_state : dict
            CPU copy of the model's state dict after local training (θ̂_r).
        last_loss : float
            Training loss on the final local gradient step.
        """
        device = self.device
        train_mask_sub = sub_batch.train_mask.squeeze(-1)

        # Initialize local model from global parameters θ̃^(t-1)
        self.load_state_dict({k: v.to(device) for k, v in epoch_init_state.items()})
        optimizer.state.clear()  # discard stale momentum from previous subgraph
        self.train()

        last_loss = 0.0
        for _ in range(self._local_epochs):
            pred = self._forward_subgraph(sub_batch)
            loss = F.binary_cross_entropy(pred[train_mask_sub], sub_batch.y[train_mask_sub])
            optimizer.zero_grad()
            self.manual_backward(loss)
            if clip_val:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm="norm",
                )
            optimizer.step()
            last_loss = loss.item()

        # Capture local parameters θ̂_r (CPU copy to free GPU memory)
        local_state = {k: v.clone().cpu() for k, v in self.state_dict().items()}
        return local_state, last_loss

    # ── Stage 1 helper: subgraph quality scoring ──────────────────────────────

    def evaluate_submodel(
        self,
        batch: Data,
        node_idx: torch.Tensor,
        val_idx_cpu: torch.Tensor,
        val_sample_size: int,
        device: torch.device,
    ) -> float:
        """Return a validation quality score s_r for the current local model.

        A random subset of validation nodes is merged into the training
        subgraph so that the GCN can use their neighbourhood context during
        inference.  The score is ``-BCE_loss`` (higher = better) and is used
        to rank subgraphs for truncation.

        Parameters
        ----------
        batch : Data
            Full-graph Data object (for neighbourhood context).
        node_idx : LongTensor
            CPU node indices of the training subgraph G_r.
        val_idx_cpu : LongTensor
            Sorted validation-node indices on CPU.
        val_sample_size : int
            Number of validation nodes to sample for the quality estimate.
        device : torch.device
            Device on which the model currently lives.

        Returns
        -------
        float
            Validation quality score s_r = −BCE_loss (higher is better).
        """
        self.eval()
        with torch.no_grad():
            # Sample a random validation mini-batch for the quality estimate
            val_sample = val_idx_cpu[torch.randperm(len(val_idx_cpu))[:val_sample_size]]
            # Augment the subgraph with validation nodes for neighbourhood context
            eval_node_idx = torch.cat([node_idx, val_sample]).unique().sort().values
            eval_sub = self._make_sub_batch(batch, eval_node_idx).to(device)

            pred_eval = self._forward_subgraph(eval_sub)

            val_local_mask = torch.isin(eval_node_idx.to(device), val_sample.to(device))
            val_loss = F.binary_cross_entropy(
                pred_eval[val_local_mask],
                eval_sub.y[val_local_mask],
            )
            score = -val_loss.item()  # negate so that higher score = better model

        self.train()
        return score

    # ── Stage 2: parameter aggregation ───────────────────────────────────────

    def aggregate_parameters(
        self,
        epoch_init_state: Dict[str, torch.Tensor],
        local_states: List[Dict[str, torch.Tensor]],
        val_scores: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate local model parameters into new global parameters θ̃^(t).

        This implements Stage 2 of the SGCN algorithm:
          1. Truncation  – discard the bottom ``truncation_ratio`` fraction of
             subgraphs ranked by their validation score s_r (top-k by score).
          2. Aggregation – compute a weighted average of the kept local
             parameter dicts θ̂_r:
               θ̃^(t) = Σ w_r * θ̂_r,   where w_r is determined by the
               ``aggregation_method`` setting:
               ``'sgcn'``     – softmax-weighted by validation score (default).
               ``'avg'``      – uniform equal-weight average.
               ``'weighted'`` – linear normalization by shifted validation score.

        Parameters
        ----------
        epoch_init_state : dict
            CPU snapshot of the global parameters at epoch start.  Used as a
            key reference to iterate over all parameter names.
        local_states : list of dict
            One CPU state dict θ̂_r per valid subgraph, in the same order as
            *val_scores*.
        val_scores : list of float
            Validation quality score s_r for each local model (higher = better).

        Returns
        -------
        dict
            Aggregated state dict θ̃^(t) (CPU tensors, same dtypes as input).
        """
        # ── Truncation: keep top (1 − truncation_ratio) subgraphs ────────────
        # n_keep is clamped to at least 1 so that aggregation always has
        # something to work with.  A truncation_ratio of 1.0 (discard all) is
        # treated as "keep the single best subgraph" rather than triggering a
        # degenerate empty aggregation.
        n_keep = max(1, int(len(local_states) * (1.0 - self._truncation_ratio)))
        kept_idx = sorted(
            range(len(val_scores)), key=lambda i: val_scores[i], reverse=True
        )[:n_keep]

        # ── Weighted aggregation of kept local parameters ─────────────────────
        kept_scores = torch.tensor([val_scores[i] for i in kept_idx], dtype=torch.float)

        if self._aggregation_method == "avg":
            weights = torch.ones(len(kept_idx), dtype=torch.float) / len(kept_idx)
        elif self._aggregation_method == "weighted":
            shifted = kept_scores - kept_scores.min() + 1e-8
            weights = shifted / shifted.sum()
        elif self._aggregation_method == "sgcn":
            weights = torch.softmax(kept_scores, dim=0)
        else:
            raise ValueError(
                f"Unknown aggregation_method: {self._aggregation_method!r}. "
                "Choose from: 'sgcn', 'avg', 'weighted'."
            )

        agg_state: Dict[str, torch.Tensor] = {}
        for key in epoch_init_state:
            stacked = torch.stack(
                [local_states[i][key].float() for i in kept_idx], dim=0
            )
            w = weights.view([-1] + [1] * (stacked.dim() - 1))
            agg_state[key] = (stacked * w).sum(dim=0).to(epoch_init_state[key].dtype)

        return agg_state

    # ── SGCN training step (orchestrates the two-stage algorithm) ─────────────

    def training_step(self, batch, *args, **kwargs):
        """One SGCN training epoch.

        Receives the full graph as *batch* (obnbench's full-batch DataLoader
        returns a single-graph Data object each call).

        The epoch is split into two explicit stages:

        Stage 1 – Local training:
          Sample R subgraphs {G_r} from the full graph (dynamic, this epoch).
          For each subgraph G_r:
            - Initialize from global parameters θ̃^(t-1)  [train_local_model]
            - Train for L gradient steps → local parameters θ̂_r  [train_local_model]
            - Score on a held-out validation mini-batch → s_r  [evaluate_submodel]

        Stage 2 – Parameter aggregation:
          Discard worst-scoring subgraphs (truncation), then compute a
          weighted average of the remaining θ̂_r to produce θ̃^(t).  [aggregate_parameters]
        """
        device = self.device
        optimizer = self.optimizers()
        clip_val = self.hparams.trainer.gradient_clip_val

        n_nodes = batch.num_nodes
        edge_index_cpu = batch.edge_index.cpu()

        # Convert boolean split masks to sorted index tensors (for samplers)
        train_idx_cpu = (
            batch.train_mask.squeeze(-1).nonzero(as_tuple=False).squeeze(1).cpu()
        )
        val_idx_cpu = (
            batch.val_mask.squeeze(-1).nonzero(as_tuple=False).squeeze(1).cpu()
        )

        # Node budget per subgraph
        if self._subgraph_max_nodes > 0:
            n_sample = min(self._subgraph_max_nodes, n_nodes)
        else:
            n_sample = max(1, int(n_nodes * self._subgraph_ratio))

        # Auto-derive subgraph count for full-graph coverage when ≤ 0
        n_subgraphs = self._n_subgraphs
        if n_subgraphs <= 0:
            n_subgraphs = math.ceil(n_nodes / n_sample)

        # Snapshot global parameters θ̃^(t-1) before any local training.
        # Every local model is initialized from this same snapshot.
        epoch_init_state = {k: v.clone().cpu() for k, v in self.state_dict().items()}

        val_sample_size = min(_SGCN_VAL_SAMPLE_SIZE, len(val_idx_cpu))

        # ── Stage 1: Local training on sampled subgraphs ─────────────────────
        # Dynamically sample R subgraphs for this epoch (coverage-aware).
        subgraphs = self.sample_subgraphs(
            batch, n_subgraphs, n_sample, edge_index_cpu, train_idx_cpu, device
        )

        local_states: List[Dict[str, torch.Tensor]] = []
        val_scores: List[float] = []
        loss_sum = 0.0

        for node_idx, sub_batch in subgraphs:
            # Initialize local model from θ̃^(t-1), train for L steps → θ̂_r
            local_state, last_loss = self.train_local_model(
                sub_batch, optimizer, epoch_init_state, clip_val
            )

            # Score local model quality on a validation mini-batch → s_r
            val_score = self.evaluate_submodel(
                batch, node_idx, val_idx_cpu, val_sample_size, device
            )

            local_states.append(local_state)
            val_scores.append(val_score)
            loss_sum += last_loss

        # ── Stage 2: Parameter aggregation ───────────────────────────────────
        if not local_states:
            # No valid subgraphs this epoch: keep global parameters unchanged
            self.load_state_dict({k: v.to(device) for k, v in epoch_init_state.items()})
            avg_loss = 0.0
        else:
            # Truncate worst subgraphs, then compute weighted average → θ̃^(t)
            agg_state = self.aggregate_parameters(epoch_init_state, local_states, val_scores)
            self.load_state_dict({k: v.to(device) for k, v in agg_state.items()})
            optimizer.state.clear()  # discard momentum from local training steps
            avg_loss = loss_sum / len(local_states)

        # ── Logging ──────────────────────────────────────────────────────────
        logger_opts = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "batch_size": n_nodes,
        }
        self.log("train/loss", avg_loss, **logger_opts)
        self.log("train/sgcn_n_valid_subgraphs", float(len(local_states)), **logger_opts)

        return avg_loss


# ─────────────────────────── GraphSAINT Lightning module ─────────────────────


class GraphSAINTModelModule(ModelModule):
    """Lightning module that trains using GraphSAINT random-walk mini-batches.

    Inherits everything from ``ModelModule`` (feature encoder, MP layers built
    from ``SGCNMPModule``, prediction head, metrics, validation / test steps,
    logging) and overrides only ``training_step`` with GraphSAINT mini-batch
    training:

    1. Build (once, lazily) a ``GraphSAINTRandomWalkSampler`` from the full
       graph received in the first ``training_step`` call.
    2. Each epoch: iterate over ``num_steps`` SAINT mini-batches.  Each
       mini-batch is an induced subgraph produced by random walks of length
       ``walk_length`` starting from ``batch_size`` randomly chosen training
       nodes.
    3. For each mini-batch: run forward, compute BCE loss on the
       ``train_mask`` nodes, and back-propagate.

    GraphSAINT hyperparameters are read from ``cfg.model.saint.*``.  See
    ``conf/model/GraphSAINT.yaml`` for default values and documentation.

    Notes
    -----
    * Uses ``automatic_optimization = False`` so the inner mini-batch loop
      is fully under our control.
    * Gradient clipping (``cfg.trainer.gradient_clip_val``) is applied
      manually via ``self.clip_gradients`` before each ``optimizer.step()``.
    * The ``GraphSAINTRandomWalkSampler`` is created once (lazy init) and
      reused across epochs for efficiency.  It is recreated automatically
      if ``num_nodes`` or ``num_edges`` of the input graph change.
    * Feature encoders that assume full-graph inputs (``AdjEmbBag``,
      ``Embedding``) are **not** compatible with subgraph training.  Use
      ``OneHotLogDeg``, ``SVD``, ``LapEigMap``, or similar encoders instead.
    """

    def __init__(self, cfg: DictConfig, node_ids: List[str], task_ids: List[str]):
        super().__init__(cfg, node_ids, task_ids)

        # Switch to manual optimization so we control the mini-batch loop
        self.automatic_optimization = False

        # Parse GraphSAINT-specific hyperparameters
        saint_cfg: Dict[str, Any] = dict(cfg.model.get("saint") or {})
        self._walk_length: int = int(saint_cfg.get("walk_length", 2))
        self._batch_size_ratio: float = float(saint_cfg.get("batch_size_ratio", 0.1))
        self._num_steps: int = int(saint_cfg.get("num_steps", 10))

        # Lazily initialized sampler (built on first training_step call)
        self._saint_loader: Optional[GraphSAINTRandomWalkSampler] = None
        self._saint_graph_sig: Optional[tuple] = None  # (num_nodes, num_edges)

    # ── forward helper (mirrors SGCNModelModule._forward_subgraph) ────────────

    def _forward_subgraph(self, sub_batch: Data) -> torch.Tensor:
        """Full model forward on *sub_batch*; returns sigmoid predictions."""
        sub_batch = self.feature_encoder(sub_batch)
        sub_batch = self.mp_layers(sub_batch)
        sub_batch = self.pred_head(sub_batch)
        return self.pred_act(sub_batch.x)  # sigmoid

    # ── sampler construction ──────────────────────────────────────────────────

    def _build_saint_loader(self, batch: Data) -> None:
        """Create and cache a ``GraphSAINTRandomWalkSampler`` from *batch*.

        *batch* is the full-graph Data object produced by obnbench's
        full-batch DataLoader.  A shallow copy is made so that the sampler
        works on CPU data without modifying the original batch.
        ``train_mask`` is squeezed to 1-D as required by PyG's sampler.
        """
        n_train = int(batch.train_mask.squeeze(-1).sum().item())
        batch_size = max(1, int(n_train * self._batch_size_ratio))

        # Shallow-copy the Data object and move to CPU so the sampler can
        # precompute its internal node-coverage statistics on CPU.
        data_cpu = copy.copy(batch.cpu())
        if data_cpu.train_mask.dim() == 2:
            data_cpu.train_mask = data_cpu.train_mask.squeeze(-1)

        self._saint_loader = GraphSAINTRandomWalkSampler(
            data_cpu,
            batch_size=batch_size,
            walk_length=self._walk_length,
            num_steps=self._num_steps,
            num_workers=0,
        )
        self._saint_graph_sig = (batch.num_nodes, batch.num_edges)

    # ── GraphSAINT training step ──────────────────────────────────────────────

    def training_step(self, batch, *args, **kwargs):
        """One GraphSAINT training epoch.

        Receives the full graph as *batch* (obnbench's full-batch DataLoader
        returns a single-graph Data each call).  The method:

        1. Lazily builds ``GraphSAINTRandomWalkSampler`` on the first call.
        2. Iterates over ``num_steps`` SAINT random-walk mini-batches.
        3. For each mini-batch: forward → BCE loss on train nodes →
           manual backward → (optional) gradient clip → optimizer step.
        4. Logs ``train/loss``.
        """
        device = self.device
        optimizer = self.optimizers()
        clip_val = self.hparams.trainer.gradient_clip_val

        # Lazy init (or rebuild if graph topology has changed)
        graph_sig = (batch.num_nodes, batch.num_edges)
        if self._saint_loader is None or self._saint_graph_sig != graph_sig:
            self._build_saint_loader(batch)

        loss_sum = 0.0
        valid_batches = 0

        self.train()
        for saint_batch in self._saint_loader:
            saint_batch = saint_batch.to(device)

            train_mask = saint_batch.train_mask
            # train_mask is already 1-D because _build_saint_loader squeezed it
            # on the source Data before constructing the sampler.
            if not train_mask.any():
                continue

            pred = self._forward_subgraph(saint_batch)
            loss = F.binary_cross_entropy(
                pred[train_mask], saint_batch.y[train_mask]
            )

            optimizer.zero_grad()
            self.manual_backward(loss)
            if clip_val:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm="norm",
                )
            optimizer.step()

            loss_sum += loss.item()
            valid_batches += 1

        avg_loss = loss_sum / valid_batches if valid_batches > 0 else 0.0

        logger_opts = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
            "batch_size": batch.num_nodes,
        }
        self.log("train/loss", avg_loss, **logger_opts)

        return avg_loss


# ─────────────────────────── builder functions ───────────────────────────────


def build_feature_encoder(cfg: DictConfig):
    feat_names = cfg.dataset.node_encoders.split("+")

    fe_list = []
    for i, feat_name in enumerate(feat_names):
        fe_cfg = cfg.node_encoder_params.get(feat_name)
        fe_cls = getattr(feature_encoders, f"{feat_name}FeatureEncoder")
        fe = fe_cls(
            dim_feat=cfg._shared.fe_raw_dims[i],
            dim_encoder=cfg.model.hid_dim,
            layers=fe_cfg.layers,
            dropout=fe_cfg.dropout,
            raw_dropout=fe_cfg.raw_dropout,
            raw_bn=fe_cfg.raw_bn,
            num_nodes=cfg._shared.num_nodes,
        )
        fe_list.append(fe)

    if len(fe_list) == 1:
        return fe_list[0]
    else:
        fe_cfg = cfg.node_encoder_params.Composed
        return feature_encoders.ComposedFeatureEncoder(
            dim_feat=cfg._shared.composed_fe_dim_in,
            dim_encoder=cfg.model.hid_dim,
            layers=fe_cfg.layers,
            dropout=fe_cfg.dropout,
            raw_dropout=fe_cfg.raw_dropout,
            raw_bn=fe_cfg.raw_bn,
            fe_list=fe_list,
        )


def build_mp_module(cfg: DictConfig):
    # ── SGCN / GraphSAINT special case: both use SGCNMPModule ───────────────
    if cfg.model.mp_type in ("SGCN", "GraphSAINT"):
        jk: bool = bool(cfg.model.get("jk") or False)
        return mp_layers.SGCNMPModule(
            dim=cfg.model.hid_dim,
            num_layers=cfg.model.mp_layers,
            dropout=cfg.model.dropout,
            jk=jk,
            mp_kwargs=dict(cfg.model.mp_kwargs or {}),
        )

    # ── Standard path ────────────────────────────────────────────────────────
    mp_cls = getattr(mp_layers, cfg.model.mp_type)
    return MPModule(
        mp_cls,
        dim=cfg.model.hid_dim,
        num_layers=cfg.model.mp_layers,
        dropout=cfg.model.dropout,
        norm_type=cfg.model.norm_type,
        act=cfg.model.act,
        act_first=cfg.model.act_first,
        residual_type=cfg.model.residual_type,
        mp_kwargs=cfg.model.mp_kwargs,
        norm_kwargs=cfg.model.norm_kwargs,
    )


def build_pred_head(cfg: DictConfig):
    return PredictionHeadModule(
        dim_in=cfg._shared.pred_head_dim_in,
        dim_out=cfg._shared.dim_out,
        num_layers=cfg.model.pred_head_layers,
        dim_inner=cfg.model.hid_dim,
    )


def build_post_proc(cfg: DictConfig):
    post_prop = None
    if cfg.model.post_prop.enable:
        post_prop = FeaturePropagation(
            num_layers=cfg.model.post_prop.num_layers,
            alpha=cfg.model.post_prop.alpha,
            norm=cfg.model.post_prop.norm,
            cached=cfg.model.post_prop.cached,
        )

    pred_act = nn.Identity() if cfg.model.skip_pred_act else nn.Sigmoid()

    post_cands = None
    if cfg.model.post_cands.enable:
        post_cands = CorrectAndSmooth(
            num_correction_layers=cfg.model.post_cands.num_correction_layers,
            num_smoothing_layers=cfg.model.post_cands.num_smoothing_layers,
            correction_alpha=cfg.model.post_cands.correction_alpha,
            smoothing_alpha=cfg.model.post_cands.smoothing_alpha,
            correction_norm=cfg.model.post_cands.correction_norm,
            smoothing_norm=cfg.model.post_cands.smoothing_norm,
            autoscale=cfg.model.post_cands.autoscale,
            scale=cfg.model.post_cands.scale,
            cached=cfg.model.post_cands.cached,
        )

    return post_prop, pred_act, post_cands
