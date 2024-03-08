import json
import math
import os
import sys
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append('data/')
sys.path.append('model/')
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import shutil
import time
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
# # pytorch lightning
from torch_geometric.datasets import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
import argparse
import numpy as np
from model.data_utils import *
import sys
from termcolor import colored
sys.path.append('model/')

from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime
import networkx as nx
import scipy as sp
import pymetis
from tqdm import tqdm
from torch_geometric.datasets import GNNBenchmarkDataset


from model.MLP import MLP
from model.PL_UniversalModel import PL_UniversalModel
import argparse
import warnings

warnings.filterwarnings("ignore")


# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments

parser.add_argument('--use_AdditionalAttr',action='store_true',default=False)
parser.add_argument('--use_KD',action='store_true',default=False)
parser.add_argument('--dataset', type=str,default="CIFAR10")
parser.add_argument('--hidden_dim', type=int,default=32)
parser.add_argument('--out_dim', type=int,default=32)
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--dropout', type=float,default=0.)
parser.add_argument('--first_layer_dropout', type=float,default=0.)
parser.add_argument('--pooling_method', type=str,default='attention')
parser.add_argument('--lr', type=float,default=8e-3)
parser.add_argument('--lr_patience', type=int,default=30)
parser.add_argument('--weight_decay', type=float,default=1e-7)
parser.add_argument('--batch_size', type=int,default=128)
parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=150)
parser.add_argument('--seed', type=int,default=1)
args = parser.parse_args()
args = vars(args)
print ("args: ", args)

torch.manual_seed(args["seed"])

dataset_name = args["dataset"]
if dataset_name== "CIFAR10":
    pre_transforms = Compose([
        SuperpixelTransform(),
        ComputeKhopNeighbors(5),
        ComputeClusteringCoefficient(),
        PerformMetisClustering(n_clusters=5),
        PerformMetisClustering(n_clusters=10),
        PerformLouvainClustering()])
else:
        pre_transforms = Compose([
        ComputeKhopNeighbors(5),
        ComputeClusteringCoefficient(),
        PerformMetisClustering(n_clusters=5),
        PerformMetisClustering(n_clusters=10),
        PerformLouvainClustering()])       

s= time()
print (colored(f"loading dataset: {dataset_name}",'red','on_yellow'))
if dataset_name == "CIFAR10" or dataset_name == "PATTERN":
    if not args['use_AdditionalAttr']:
        train_dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='train',name=dataset_name)
        val_dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='val',name=dataset_name)
        test_dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='test',name=dataset_name)
    else:
        train_dataset = GNNBenchmarkDataset(root=f'data/withAdditionalAttr/',split='train',name=dataset_name)
        val_dataset = GNNBenchmarkDataset(root=f'data/withAdditionalAttr/',split='val',name=dataset_name)
        test_dataset = GNNBenchmarkDataset(root=f'data/withAdditionalAttr/',split='test',name=dataset_name)  

    
    # for test
    # train_dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='test',name=dataset_name)
    # val_dataset = train_dataset
    # test_dataset = train_dataset
t = time()
print (t-s, f"seconds used to load dataset {dataset_name}")
print (test_dataset[0])


if dataset_name == "CIFAR10":
    args['num_classes'] = 10
if dataset_name == "PATTERN":
    args['num_classes'] = 2


args['node_dim'] = train_dataset.num_features
mlp = MLP(**args)
args["model"] = mlp

# for real-use
train_dloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=4 if torch.cuda.is_available() else 0)
valid_dloader = DataLoader(val_dataset,batch_size=500,shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
test_dloader = DataLoader(test_dataset,batch_size=500,shuffle=False,num_workers=4 if torch.cuda.is_available() else 0)
args['test_loader'] = test_dloader


# for test
# test_dloader = DataLoader(dataset[:60],batch_size=32,shuffle=False,num_workers=8 if torch.cuda.is_available() else 0)
# args['val_loader'] = test_dloader
# args['test_loader'] = test_dloader


pl_model = PL_UniversalModel(**args)
print ("model loaded")
if torch.cuda.is_available():
    device = get_free_gpu(10) # more than 10GB free GPU memory
else:
    device = None
if args['device_id']>=0:
    device = args['device_id']
print (colored(f"using device: {device}",'red','on_yellow'))


trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/MLP/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,enable_checkpointing=False)
trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=valid_dloader)


