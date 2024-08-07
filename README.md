# MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification

This repository contains codes for our work __MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification__ accepted by TheWebConf 2024. 

In this work, we propose a distillation framework to bridge the gap between efficiency and expressiveness of graph neural networks. MuGSI achieves the same performance as 3-WL GNN and GIN in most of datasets, and is 10x-100x more memory efficient than 3-WL GNNs, and robust to test-time distributional shift.

## Set up environment

To set up the virtual environment for our project, use _requirements.txt_ to install related libraries.

## Load and pre-process datasets

To load and preprocess TUdatasets, use the following command:

```bash
python DataPreTransform_TU.py --dataset $dataset_name$
```

To load and preprocess TUdatasets CIFAR10 and MOLHIV datasets, use the following command:

```bash
python DataPreTransform.py --dataset $dataset_name$
```

## Download pre-trained GNN models

To download the pre-trained GNN models as teachers, go to [this link](https://drive.google.com/file/d/1QYlPVbzJua4Ql5PBIl2wtciB2f3f7bnb/view?usp=drive_link) to download the pre-trained GNN models, unzip it and put the `best_models/` under the root directory.

## Run MuGSI

To run MuGSI, here is an example that uses LaPE, the student model is MLP, teacher model is GIN, the hyper-parameters to control graph-distillation, cluster distillation and node distillation are indicated by `--graphPoolingReg`, `--ClusterMatchingReg` and `--RandomWalkConsistencyReg` respectively. 

```bash
python run_Model_TU.py --use_KD  --device_id 0 --max_epochs 350 --dataset REDDIT-BINARY --hidden_dim 64 --out_dim 64 --dataset_index 0 --studentModelName MLP --teacherModelName GIN --lr_patience 30 --usePE --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --useRandomWalkConsistency --RandomWalkConsistencyReg 0.0001 --useClusterMatching --ClusterMatchingReg 0.01 --useGraphPooling --graphPoolingReg 0.01 --KD_name MuGSI
```

To use GA-MLP as student model, use `--studentModelName GA-MLP`, remove `--usePE` to disable LaPE. `--dataset_index` indicates the fold used for the dataset, ranging from 0-9.

Tp run MolHiv, use the following command:

```bash
python run_Model_OGB.py  --device_id 0 --trialId 1 --use_KD --dataset ogbg-molhiv  --drop_ratio 0.5  --studentModelName GA-MLP --lr_patience 30  --numWorkers 4 --useSoftLabel --softLabelReg 1.0 --useRandomWalkConsistency --RandomWalkConsistencyReg 0.0001 --useClusterMatching --ClusterMatchingReg 0.01 --useGraphPooling --graphPoolingReg 0.01 --KD_name useJoint
```

Similarly, one can use different student model, and try with different hyper-parameters for _RandomWalkConsistencyReg_, _ClusterMatchingReg_ and _graphPoolingReg_.

To use 3-WL GNN (KPGIN) as teacher model for TUDatasets, run the following command:

```bash
python run_Model_TU_KPGNN.py --use_KD --hidden_size 64 --drop_prob 0.5 --K 4 --kernel spd --combine geometric --num_layer 3 --max_epochs 350  --dataset IMDB-BINARY  --dataset_index 0 --studentModelName GA-MLP --teacherModelName KPGIN  --batch_size 32  --numWorkers 4  --useSoftLabel --softLabelReg 1.0 --useRandomWalkConsistency --RandomWalkConsistencyReg 0.0001 --useClusterMatching --ClusterMatchingReg 0.01 --useGraphPooling --graphPoolingReg 0.01 --KD_name useJoint
```

The backbone architecture can be tuned via: _hidden_size_, _K_, _combine_ and _num_layer_. More details can be found in our paper.

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

# Citation

If you find our paper and repo useful, please cite our paper:

```bibtex
@inproceedings{Yao_2024, series={WWW ’24},
   title={MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification},
   url={http://dx.doi.org/10.1145/3589334.3645542},
   DOI={10.1145/3589334.3645542},
   booktitle={Proceedings of the ACM on Web Conference 2024},
   publisher={ACM},
   author={Yao, Tianjun and Sun, Jiaqi and Cao, Defu and Zhang, Kun and Chen, Guangyi},
   year={2024},
   month=may, collection={WWW ’24} }
```
