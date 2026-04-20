"""obnbench/model_layers/mp_layers.py  – drop-in replacement for the original file.

Changes vs. upstream krishnanlab/obnbench:
  • Added ``SGCNConv``     – thin GCNConv wrapper with no self-loops (matches the
                             original SGCN model).
  • Added ``SGCNMPModule`` – standalone nn.Module implementing SGCN's layerwise
                             residual-skip (pre-BN) and optional Jumping-Knowledge
                             sum aggregation.  It is used by ``build_mp_module``
                             in model.py when ``cfg.model.mp_type == 'SGCN'``.
"""

import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

# TODO: GAT/GINE edge attr mod bining


# ─────────────────────────── original convolution wrappers ───────────────────


class BaseConvMixin:

    _edge_usage = "none"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_edge_feature: bool = True,
        **kwargs,
    ):
        # Set up forward function depending on the edge usage capabilities
        # of the wrapped convolution module.
        # Use simple (no-edge) forward when: the conv does not support edges,
        # OR the caller explicitly opts out of edge features.
        if self._edge_usage == "none" or not use_edge_feature:
            self._forward = self._forward_simple
        elif self._edge_usage == "edge_weight":
            self._forward = self._forward_edgeweight
        elif self._edge_usage == "edge_attr":
            self._forward = self._forward_edgeattr
        else:
            raise ValueError(
                f"Unknown edge usage mode {self._edge_usage!r}, "
                "available options are: 'none', 'edge_weight', 'edge_attr'",
            )

        super().__init__(in_channels, out_channels, **kwargs)

    def _forward_simple(self, batch):
        return super().forward(batch.x, batch.edge_index)

    def _forward_edgeweight(self, batch):
        return super().forward(batch.x, batch.edge_index, edge_weight=batch.edge_weight)

    def _forward_edgeattr(self, batch):
        if (edge_attr := batch.edge_attr) is None:
            warnings.warn(
                "Implicitly use edge_attr in place of edge_weight because "
                "edge_attr is unavailable.",
                stacklevel=2,
            )
            # Try to use edge weight as attr if edge attr is unavailable
            edge_attr = batch.edge_weight
        return super().forward(batch.x, batch.edge_index, edge_attr=edge_attr)

    def forward(self, batch):
        batch.x = self._forward(batch)
        return batch


class GATConv(BaseConvMixin, pygnn.GATConv):

    _edge_usage = "edge_attr"


class GATv2Conv(BaseConvMixin, pygnn.GATv2Conv):

    _edge_usage = "edge_attr"


class GCNConv(BaseConvMixin, pygnn.GCNConv):

    _edge_usage = "edge_weight"


class GENConv(BaseConvMixin, pygnn.GENConv):

    _edge_usage = "edge_weight"


class PatchedGINConv(pygnn.GINConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        hidden_channels: Optional[int] = None,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs,
    ):
        hidden_channels = hidden_channels or out_channels
        mlp = pygnn.MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act="relu",
            norm="batch_norm",
        )
        super().__init__(mlp, eps=eps, train_eps=train_eps, **kwargs)


class GINConv(BaseConvMixin, PatchedGINConv):

    _edge_usage = "none"


class PatchedGINEConv(pygnn.GINEConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_edge_feature: bool = True,
        *,
        num_layers: int = 2,
        hidden_channels: Optional[int] = None,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        hidden_channels = hidden_channels or out_channels
        mlp = pygnn.MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act="relu",
            norm="batch_norm",
        )
        super().__init__(mlp, eps=eps, train_eps=train_eps, edge_dim=edge_dim, **kwargs)


class GINEConv(BaseConvMixin, PatchedGINEConv):

    _edge_usage = "edge_attr"


class SAGEConv(BaseConvMixin, pygnn.SAGEConv):

    _edge_usage = "none"


class GatedGCNConv(BaseConvMixin, pygnn.ResGatedGraphConv):

    _edge_usage = "none"


# ─────────────────────────── SGCN-specific additions ─────────────────────────


class SGCNConv(BaseConvMixin, pygnn.GCNConv):
    """GCN convolution layer as used by SGCN.

    Identical to ``GCNConv`` above but sets ``add_self_loops=False`` by default
    (matching the original SGCN implementation) and always uses
    ``edge_weight`` when available.
    """

    _edge_usage = "edge_weight"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_edge_feature: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("add_self_loops", False)
        kwargs.setdefault("normalize", True)
        super().__init__(in_channels, out_channels, use_edge_feature, **kwargs)


class SGCNMPModule(nn.Module):
    """Message-passing module implementing SGCN's residual-skip + optional JK.

    This module is a direct adaptation of ``GNN_PyG`` from the SGCN codebase,
    rewritten to operate on obnbench's PyG ``Data`` batch objects (using scalar
    ``edge_weight`` instead of multi-dimensional ``edge_attr``).

    Architecture per layer
    ----------------------
    ::

        h_conv      = SGCNConv(h_in)            # GCN conv, no self-loops
        h           = h_conv + h_last_pre_bn    # pre-BN residual skip
        h_last_pre_bn = h                       # save for next layer
        h           = BatchNorm1d(h)
        h           = ReLU(h)
        h           = Dropout(h)

    When ``jk=True``, per-layer post-activation outputs are summed at the end
    (Jumping Knowledge – sum aggregation).

    Parameters
    ----------
    dim : int
        Uniform hidden dimension (in = out = dim for every layer).
    num_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability applied after each activation.
    use_edge_feature : bool
        If True, ``edge_weight`` is passed to the GCN convolution.
    jk : bool
        Enable Jumping Knowledge sum aggregation over all layer outputs.
    mp_kwargs : dict, optional
        Extra keyword arguments forwarded to ``SGCNConv`` (e.g. ``bias``).

    Notes
    -----
    This module bypasses ``MPModule`` entirely and is returned directly from
    ``build_mp_module`` when ``cfg.model.mp_type == 'SGCN'``.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        dropout: float = 0.0,
        use_edge_feature: bool = True,
        jk: bool = False,
        mp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.jk = jk

        _mp_kwargs: Dict[str, Any] = dict(mp_kwargs or {})
        _mp_kwargs.setdefault("add_self_loops", False)
        _mp_kwargs.setdefault("normalize", True)

        self.convs = nn.ModuleList(
            [SGCNConv(dim, dim, use_edge_feature, **_mp_kwargs) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, batch):
        h_last: Optional[torch.Tensor] = None
        h_local = []

        for conv, norm in zip(self.convs, self.norms):
            batch = conv(batch)   # updates batch.x via SGCNConv / GCNConv
            h = batch.x

            # Pre-BN residual skip: add raw conv output of previous layer
            if h_last is not None:
                h = h + h_last[: h.shape[0], :]
            h_last = h  # save pre-BN representation for next layer

            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            h_local.append(h)
            batch.x = h

        # Jumping Knowledge: sum all per-layer post-activation outputs
        if self.jk and len(h_local) > 1:
            h_local_trimmed = [t[: batch.x.shape[0], :] for t in h_local]
            batch.x = torch.stack(h_local_trimmed, dim=0).sum(dim=0)

        return batch


# ─────────────────────────────── public API ──────────────────────────────────

__all__ = [
    "GATConv",
    "GATv2Conv",
    "GCNConv",
    "GENConv",
    "GINConv",
    "GINEConv",
    "SAGEConv",
    "GatedGCNConv",
    # SGCN additions
    "SGCNConv",
    "SGCNMPModule",
]


if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    m = GATConv(2, 5)
    data = Data(
        x=torch.ones(4, 2),
        edge_index=torch.LongTensor([[1, 2, 3], [0, 0, 0]]),
    )

    print(f"{m=}")
    print(f"{data=}")
    print(f"{m.forward(data)=}")
