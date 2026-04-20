# Benchmarking repository for the Open Biomedical Network Benchmark

This is a benchmarking repository accompanying the [`obnb`](https://github.com/krishnanlab/obnb) Python package.

## Set up environment

```bash
conda create -n obnb python=3.8 -y && conda activate obnb

# Install PyTorch and PyG with CUDA 11.7
conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.3.0 -c pyg -y

pip install obnb[ext]==0.1.0  # install obnb with extension modules (PecanPy, GraPE, ...)
pip install -r requirements_extra.txt  # extra dependencies for benchmarking

conda clean --all -y  # clean up
```

The extra dependencies are, e.g.,

- [`Hydra`](https://github.com/facebookresearch/hydra) for managing experiments.
- [`Lightning`](https://lightning.ai/docs/pytorch/latest/) for organizing model training framework.
- [`WandB`](https://docs.wandb.ai/) for logging metrics.

**Note**: if you do not need to run the benchmarking experiments and only want to play around
with our benchmarking results with one of the [notebooks](notebook), you can skip the installation
for PyTorch and PyG.

```bash
pip install obnb[ext]==0.1.0
```

## Set up data (optional)

Run `get_data.py` to download and set up data for all the experiments.
Data will be saved under the `datasets/` directory by default, and will take up approximately 6 GB of space.

```bash
python get_data.py
```

This step is completely optional and directly runing the training script will work fine.
But runing `get_data.py` once before training prevents multiple parallel jobs doing the same data preprocessing
work if the processed data is not available yet.

## Run experiments

After setting up the data, one can run a single experiment by specifying the choices of network, label, and model:

```bash
python main.py dataset.network=BioGRID dataset.label=DisGeNET model=GCN
```

Check out the [`conf/model/`](conf/model) directory for all available model presets.
The main model presets are:

- `GCN`
- `GAT`
- `GCN+BoT`
- `GAT+BoT`
- `LogReg+Adj`
- `LogReg+Node2vec`
- `LogReg+Walklets`

### Run batch of parallel jobs

```bash
cd run

# GNN node feature ablation (example of runing GCN with node2vec features on BioGRID)
sh run_abl_gnn_feature.sh GCN BioGRID Node2vec

# C&S ablation (example of runing GCN with C&S post processing on BioGRID)
sh run_abl_cs.sh GCN BioGRID

# GNN label reuse ablation (example of runing GCN with label reuse on BioGRID)
sh run_abl_gnn_label.sh GCN BioGRID

# GNN label reuse with C&S ablation (example of runing GCN with label reuse with C&S on BioGRID)
sh run_abl_gnn_cs_label.sh GCN BioGRID

# GNN with bag of tricks, i.e., node2vec node feature + label reuse + C&S
sh run_gnn_bot.sh GCN BioGRID
```

To run all experiments presented in the paper (may take several days):

```bash
sh run_all.sh
```

### Tuning with W&B

First create a sweep agent, e.g., for BioGRID-DisGeNET-GCN:

```bash
wandb sweep conf/tune/BioGRID-DisGeNET-GCN.yaml
```

Then, follow the instruction from the command above to spawn sweep agents to automatically
tune the model configuration on a particular dataset.

## Results anallysis

To run the [notebooks](notebook), first download our benchmarking results
(or you can rerun all the benchmarking experiments yourself using our run scripts described above).
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8048305.svg)](https://doi.org/10.5281/zenodo.8048305)

```bash
wget -O results/main.csv.gz https://zenodo.org/record/8048305/files/main.csv.gz
```

## Data stats (`obnbdata-0.1.0`) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8045270.svg)](https://doi.org/10.5281/zenodo.8045270)

### Networks

| Network | Weighted | Num. nodes | Num. edges | Density | Category |
| :------ | :------: | ---------: | ---------: | ------: | -------: |
| [HumanBaseTopGlobal](https://humanbase.net/) | :white_check_mark: | 25,689 | 77,807,094 | 0.117908 | Large & Dense |
| [STRING](https://string-db.org/) | :white_check_mark: | 18,480 | 11,019,492 | 0.032269 | Large |
| [ConsensusPathDB](http://cpdb.molgen.mpg.de/) | :white_check_mark: | 17,735 | 10,611,416 | 0.033739 | Large |
| [BioGRID](https://thebiogrid.org/) | :x: | 19,765 | 1,554,790 | 0.003980 | Medium |
| [SIGNOR](https://signor.uniroma2.it/) | :x: | 5,291 | 28,676 | 0.001025 | Small |

### Labels

#### [DISEASES](https://diseases.jensenlab.org/About)

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 145 | 178.1 | 137.4 | 127.0 |
| ConsensusPathDB | 144 | 177.4 | 137.5 | 126.0 |
| HumanBaseTopGlobal | 149 | 178.5 | 137.7 | 129.0 |
| SIGNOR | 89 | 144.6 | 89.4 | 117.0 |
| STRING | 146 | 175.4 | 135.6 | 126.0 |

#### [DisGeNET](https://www.disgenet.org/)

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 305 | 208.3 | 143.1 | 159.0 |
| ConsensusPathDB | 298 | 207.4 | 140.8 | 161.5 |
| HumanBaseTopGlobal | 287 | 219.7 | 145.7 | 173.0 |
| SIGNOR | 219 | 147.3 | 81.9 | 124.0 |
| STRING | 296 | 208.0 | 140.6 | 162.0 |

#### [GOBP](http://geneontology.org/)

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 114 | 89.5 | 37.1 | 76.0 |
| ConsensusPathDB | 112 | 90.1 | 37.0 | 76.5 |
| HumanBaseTopGlobal | 115 | 89.2 | 37.3 | 76.0 |
| SIGNOR | 41 | 81.3 | 22.7 | 78.0 |
| STRING | 116 | 88.9 | 36.6 | 75.0 |
