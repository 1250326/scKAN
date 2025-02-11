# scKAN: Kolmogorov-Arnold Network for GRN Inference

This repository contains the official implementation of **Kolmogorov-Arnold Network for Gene Regulatory Network Inference**.

## Setup
### Environment
Please note that the following is only a record of our environment, and we believe our code can adapt to a wide range of versions.
- python: 3.9.18
- shap: 0.45.1
- pytorch: 2.2.1
- pytorch-cuda: 11.8
- numpy: 1.23.5
- pandas: 2.2.1

Package `efficient-kan` are downloaded from [https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan). 
<!-- commit 7b6ce1c87f18c8bc90c208f6b494042344216b11 -->

### Data source
We used the BEELINE benchmark, and their datasets can be downloaded from [https://zenodo.org/records/3701939](https://zenodo.org/records/3701939).

BEELINE datasets contain 6 synthetic networks and 4 curated networks.
BEELINE has provided 10 simulated datasets for each network using BoolODE.

We also generated Cellectri datasets with 71 and 104 genes using BoolODE, which can be found in `sample_data`.

All example datasets (e.g. GSD-2000-1) provided in `sample_data` contain the following:

```
- GSD-2000-1
  |- ExpressionData.csv
  |- PseudoTime.csv
  |- refNetwork.csv
```
In our experiment, we only used `ExpressionData.csv` for training and `refNetwork.csv` for evaluation.
Random seed is automatically set to the 2nd number of the dataset name, such as `1` for `GSD-2000-1`.

## Training
You can reproduce our results via:
```bash
save_folder="logs/"
dataset_path="sample_data/GSD-2000-1"

python KAN.py --dataset-path $dataset_path --save-path $save_folder"
```

## Evaluation
We use three metrics for evaluation:
- AUROC: This is calculated with `sklearn.metrics.roc_auc_score`
- AUPRC: This is calculated with `sklearn.metrics.average_precision_score`

A sample evaluation script `eval.py` is provided.

Usage:
```bash
save_folder="logs/GSD-2000-1"
ref_net="sample_data/GSD-2000-1/refNetwork.csv"

python eval.py --exp-dir $save_folder --ref-net-path $ref_net
```

<!-- ## Citation
TBD -->