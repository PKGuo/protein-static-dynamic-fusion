# Boosting Protein Graph Representations through Static-Dynamic Fusion

This repository contains the code for the paper "Boosting Protein Graph Representations through Static-Dynamic Fusion" accepted at ICML 2025.

## Overview

We propose a novel heterogeneous graph representation that integrates both static structural information and dynamic correlations from molecular dynamics trajectories for protein modeling. Our approach applies relational graph neural networks to process these enriched representations.

## Environment Setup

This codebase is adapted from [MISATO-affinity](https://github.com/kierandidi/misato-affinity). To create the conda environment:

```bash
# For GPU environment
conda env create -f requirements/env_gpu.yml
conda activate <env_name>
```

## Data Preparation

The required data needs to be downloaded and preprocessed. Please refer to [`preprocessing/PREPROCESSING_GUIDE.md`](preprocessing/PREPROCESSING_GUIDE.md) for detailed instructions.

## Training

Example training command:

```bash
python src/train.py \
    graph_type=combined \
    model=rgcn/rgcn_node_regressor \
    datamodule=atom_adaptability_datamodule_aligned \
    logger=wandb_adaptability
```

## Citation

If you found this work useful, please consider citing our paper:

```bibtex
@inproceedings{
guo2025boosting,
title={Boosting Protein Graph Representations through Static-Dynamic Fusion},
author={Pengkang Guo and Bruno Correia and Pierre Vandergheynst and Daniel Probst},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=QbtBIE36Fd}
}
```

