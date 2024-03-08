import json
import math
import os
import sys
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
from termcolor import colored

from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime
import networkx as nx
import scipy as sp
import pymetis
from tqdm import tqdm
from torch_geometric.datasets import GNNBenchmarkDataset
from model.training_utils import *
from model.training_utils import NaNStopping

from model.MLP import MLP,GA_MLP
from model.base_GNN_models import GIN
# from model.base_GNN_models import GCN
from model.PL_UniversalModel_TU import PL_UniversalModel_TU
from model.data_cache import DatasetVars
import argparse
import warnings
from glob import glob

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments

parser.add_argument('--dataset', type=str,default="IMDB-MULTI")
parser.add_argument('--dataset_index', type=int,default=0)
parser.add_argument('--num_hops', type=int,default=1)
parser.add_argument('--use_edge_feats',action='store_false',default=True)
parser.add_argument('--hidden_dim', type=int,default=32)
parser.add_argument('--num_layers', type=int,default=4)
parser.add_argument('--out_dim', type=int,default=32)
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--dropout', type=float,default=0.)
parser.add_argument('--first_layer_dropout', type=float,default=0.)
parser.add_argument('--pooling_method', type=str,default='attention')
parser.add_argument('--Khop_pooling_method', type=str,default='attention')
parser.add_argument('--lr', type=float,default=8e-3)
parser.add_argument('--lr_patience', type=int,default=30)
parser.add_argument('--weight_decay', type=float,default=0.0)
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=300)
parser.add_argument('--numWorkers', type=int,default=2)


parser.add_argument('--use_KD',action='store_true',default=False)
parser.add_argument('--KD_name', type=str,default="NULL")
parser.add_argument('--studentModelName', type=str,default="MLP")
parser.add_argument('--teacherModelName', type=str,default="GIN")
parser.add_argument('--teacherFileIndex', type=int,default=0)
parser.add_argument('--useSoftLabel',action='store_true',default=False)
parser.add_argument('--useNodeSim',action='store_true',default=False)
parser.add_argument('--useNodeFeatureAlign',action='store_true',default=False)
parser.add_argument('--useClusterMatching',action='store_true',default=False)
parser.add_argument('--clusterAlgo', type=str,default="louvain")
parser.add_argument('--useRandomWalkConsistency',action='store_true',default=False)
parser.add_argument('--useDropoutEdge',action='store_true',default=False)
parser.add_argument('--useMixUp',action='store_true',default=False)
parser.add_argument('--useGraphPooling',action='store_true',default=False)
parser.add_argument('--useNCE',action='store_true',default=False)

parser.add_argument('--softLabelReg', type=float,default=1e-1)
parser.add_argument('--nodeSimReg', type=float,default=1e-3)
parser.add_argument('--NodeFeatureReg', type=float,default=1e-1)
parser.add_argument('--ClusterMatchingReg', type=float,default=1e-1)
parser.add_argument('--RandomWalkConsistencyReg', type=float,default=1e-1)
parser.add_argument('--graphPoolingReg', type=float,default=1e-1)
parser.add_argument('--pathLength', type=int,default=8)
parser.add_argument('--MixUpReg', type=float,default=0.)
parser.add_argument('--NCEReg', type=float,default=0.)

parser.add_argument('--seed', type=int,default=1)


# for additional attributes
parser.add_argument('--use_AdditionalAttr',action='store_false',default=True)
parser.add_argument('--usePE',action='store_true',default=False)


args = parser.parse_args()
args = vars(args)


#! watch out this, this is uncomment when testing
# args["useRandomWalkConsistency"]=True
# args["use_AdditionalAttr"]=True
# args["useNCE"]=True

# args["use_KD"]=False
# args["studentModelName"]="SKGCN"

# args["useGraphPooling"]=True
# args["useClusterMatching"]=True


for k in args:
    print (colored(f"{k}: {args[k]}",'red','on_white'))

if torch.cuda.is_available():
    device = get_free_gpu(9) # more than 10GB free GPU memory
else:
    device = None
if args['device_id']>=0:
    device = args['device_id']
print (colored(f"using device: {device}",'red','on_yellow'))
args['device_id'] = device

torch.manual_seed(args["seed"])
random_id = int(np.random.choice(5000000)) # only for showing exp
args["random_id"] = random_id

dataset_name = args["dataset"]
args['dataset_name'] = dataset_name

t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
args['num_classes'] = t['num_classes']
if args["usePE"]:
    args["node_dim"] = t["num_features"]+20
else:
    args["node_dim"] = t["num_features"]


model_saving_path = f"best_models/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}/fold_index_{args['dataset_index']}/"
loss_saving_path = f"LossCurves/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["loss_saving_path"] = loss_saving_path

if args['use_KD']:
    result_saving_path = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/pathLength_{args['pathLength']}_hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_nodeSim_{args['nodeSimReg']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
    args["model_saving_path"] = model_saving_path
    args["result_saving_path"] = result_saving_path

else:
    if args["studentModelName"] == "GIN" or args["studentModelName"] == "GCN":
        args["model_saving_path"] = model_saving_path
    else:
        args["model_saving_path"] = None
    args["result_saving_path"] = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_noKD/fold_index_{args['dataset_index']}/"


if args['studentModelName'] == "MLP":
    print (colored(f"using MLP as student model",'red','on_yellow'))
    student_model = MLP(**args)
    args["model"] = student_model


elif args['studentModelName'] == "GIN":
    print (colored(f"using GIN as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/raw/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    student_model = GIN(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GCN":
    print (colored(f"using GCN as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/raw/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    student_model = GCN(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GA-MLP":
    print (colored(f"using GA-MLP as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/GA_MLP/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    args['edge_dim'] = 10
    if "edge_attr" in tmp_dataset[0]:
        args['edge_dim'] = tmp_dataset[0].edge_attr.shape[1]
    student_model = GA_MLP(**args)
    args["model"] = student_model

if args["use_KD"]:
    raw_dataset = TUDataset(root=f"data/raw/",name=dataset_name)
    t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
    hidden,dropout,num_layers,num_classes = t["hidden"],t["dropout"],t["num_layers"],t["num_classes"]
    args["teacher_hidden_dim"] = hidden
    if args['teacherModelName'] == "GIN":
        teacherModel = GIN(hidden_dim=hidden,dropout=dropout,num_layers=num_layers,num_classes=num_classes,pyg_dataset=raw_dataset,dataset_name=dataset_name)
        TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
    elif args['teacherModelName'] == "GCN":
        teacherModel = GCN(hidden_dim=hidden,dropout=dropout,num_layers=num_layers,num_classes=num_classes,pyg_dataset=raw_dataset,pooling_method="attention",dataset_name=dataset_name)
        TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"      
        print (colored(f"teacherModel:{TmodelPath}",'red'))
    
    print (TmodelPath)
    TmodelPath = glob(os.path.join(TmodelPath,'*.pt'))[0]
    print (f"teacherModel path:{TmodelPath}")
    if device>=0:
        # qq = torch.load(TmodelPath,map_location=torch.device(f"cuda:{device}"))
        # print (qq["pred.weight"].shape)
        teacherModel.load_state_dict(torch.load(TmodelPath,map_location=torch.device(f"cuda:{device}")))
    else:
        assert "No Free GPUs"
    teacherModel.eval()

    transforms = []
    if args['useDropoutEdge']:
        transforms.append(DropEdge(p=0.07))
    if args['useRandomWalkConsistency']:
        transforms.append(RandomPathTransform(path_length=args["pathLength"]))
    if args['useMixUp']:
        args['teacherModel']=teacherModel
    

    transforms.append(TeacherModelTransform(teacherModel,use_clustering=args["useClusterMatching"],cluster_algo=args["clusterAlgo"]))
    transforms = Compose(transforms)
    
    
s= time()
print (colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

dataset_idx = args['dataset_index']
dataset_path = f'data/withAdditionalAttr/' if args['studentModelName'] != "GA-MLP" else f'data/GA_MLP/'
if args["use_KD"]:
    tu_dataset = TUDataset(root=dataset_path,name=dataset_name,transform=transforms)
else:
    tu_dataset = TUDataset(root='data/raw/',name=dataset_name)
train_indices, test_indices = k_fold_without_validation(tu_dataset,10)
train_indices = train_indices[dataset_idx]
test_indices = test_indices[dataset_idx]
train_dataset = tu_dataset[train_indices]
test_dataset = tu_dataset[test_indices]
args["pyg_dataset"] = test_dataset


t = time()
print (t-s, f"seconds used to load dataset {dataset_name}")
print (test_dataset[0])

# for real-use
train_dloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=args["numWorkers"])
# valid_dloader = DataLoader(val_dataset,batch_size=500,shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
test_dloader = DataLoader(test_dataset,batch_size=100,shuffle=False,num_workers=args["numWorkers"])
args['test_loader'] = test_dloader


# for test
# test_dloader = DataLoader(dataset[:60],batch_size=32,shuffle=False,num_workers=8 if torch.cuda.is_available() else 0)
# args['val_loader'] = test_dloader
# args['test_loader'] = test_dloader

pl_model = PL_UniversalModel_TU(**args)

# early stop
early_stop_callback = EarlyStopping(
   monitor='valid_acc',
   min_delta=0.00,
   patience=60,
   verbose=False,
   mode='max'
)

if args["use_KD"]:
    trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/{args["studentModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),early_stop_callback],enable_checkpointing=False)
    trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)
else:
    trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/{args["studentModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),early_stop_callback],enable_checkpointing=False)
    trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)    
torch.cuda.empty_cache()


