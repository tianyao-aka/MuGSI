# MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification

This repository contains a subset of codes for our submission: _MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification_ to WWW2024. The codes contain the main results in Table 2 for TUDatasets, using GIN as the teacher model, and MLP or GA-MLP as the student model. We will make **all codes publicly available once accepted**.

## Set up environment

To set up the virtual environment for our project, use _requirements.txt_ to install related libraries.

## Load and pre-process datasets

To load and preprocess datasets, use the following command:

```bash
python DataPreTransform_TU.py --dataset $dataset_name$
```

## Download pre-trained GNN models

To download the pre-trained GNN models as teachers, go to [this link](https://drive.google.com/file/d/1QYlPVbzJua4Ql5PBIl2wtciB2f3f7bnb/view?usp=drive_link) to download the pre-trained GNN models, unzip it and put the `best_models/` under the root directory.

## Run MuGSI

To run MuGSI, here is an example that uses LaPE, the student model is MLP, teacher model is GIN, the hyper-parameters to control graph-distillation, cluster distillation and node distillation are indicated by `--graphPoolingReg`, `--ClusterMatchingReg` and `--RandomWalkConsistencyReg` respectively. 

```bash
python run_Model_TU.py --use_KD  --device_id 0 --max_epochs 350 --dataset REDDIT-BINARY --hidden_dim 64 --out_dim 64 --dataset_index 0 --studentModelName MLP --teacherModelName GIN --lr_patience 30 --usePE --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --useRandomWalkConsistency --RandomWalkConsistencyReg 0.0001 --useClusterMatching --ClusterMatchingReg 0.01 --useGraphPooling --graphPoolingReg 0.01 --KD_name MuGSI
```

To use GA-MLP as student model, use `--studentModelName GA-MLP`, remove `--usePE` to disable LaPE. `--dataset_index` indicates the fold used for the dataset, ranging from 0-9.

## Run GLNN

To run GLNN, only use `--useSoftLabel`, here is an example:
```bash
python run_Model_TU.py --use_KD  --device_id 0 --max_epochs 350 --dataset REDDIT-BINARY --hidden_dim 64 --out_dim 64 --dataset_index 0 --studentModelName MLP --teacherModelName GIN --lr_patience 30 --usePE --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --KD_name GLNN
```

## Run NOSMOG

To run NOSMOG, see the following example:

```bash
python run_Model_TU.py --use_KD  --device_id 0 --max_epochs 350 --dataset REDDIT-BINARY --hidden_dim 64 --out_dim 64 --dataset_index 0 --studentModelName MLP --teacherModelName GIN --lr_patience 30 --usePE --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --useNodeSim --nodeSimReg 0.1  --KD_name useNOSMOG
```

## Overall Framework

<p align="center">
  <br />
  <img src="Model.png" width="800">
  <br />
</p>

## Results

<p align="center">
  <br />
  <img src="results.png" width="800">
  <br />
</p>



