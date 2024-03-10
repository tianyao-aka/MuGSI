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

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from model.GIN import GNN,MLP,GA_MLP
from model.PL_UniversalModel_OGBMOL import PL_UniversalModel
from model.data_cache import DatasetVars
import argparse
import warnings
print (pl.__version__)
warnings.filterwarnings("ignore")


# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments


parser.add_argument('--dataset', type=str,default="ogbg-molpcba")
parser.add_argument('--num_hops', type=int,default=1)
parser.add_argument('--hidden_dim', type=int,default=300)
parser.add_argument('--num_layer', type=int,default=4)
parser.add_argument('--out_dim', type=int,default=300)
parser.add_argument('--num_classes', type=int,default=128)
parser.add_argument('--drop_ratio', type=float,default=0.5)
parser.add_argument('--graph_pooling', type=str,default='sum')
parser.add_argument('--lr', type=float,default=1e-3)
parser.add_argument('--lr_patience', type=int,default=15)
parser.add_argument('--weight_decay', type=float,default=0.0)
parser.add_argument('--batch_size', type=int,default=96)
parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=100)
parser.add_argument('--numWorkers', type=int,default=2)
# for KD arguments
parser.add_argument('--use_KD',action='store_true',default=False)
parser.add_argument('--KD_name', type=str,default="NULL")
parser.add_argument('--studentModelName', type=str,default="GA-MLP")
parser.add_argument('--teacherModelName', type=str,default="GIN")
parser.add_argument('--teacherFileIndex', type=int,default=0)
parser.add_argument('--useSoftLabel',action='store_true',default=True)
parser.add_argument('--useNodeSim',action='store_true',default=False)
parser.add_argument('--useNodeFeatureAlign',action='store_true',default=False)
parser.add_argument('--useClusterMatching',action='store_true',default=True)
parser.add_argument('--clusterAlgo', type=str,default="louvain")
parser.add_argument('--useRandomWalkConsistency',action='store_true',default=True)
parser.add_argument('--useDropoutEdge',action='store_true',default=False)
parser.add_argument('--useMixUp',action='store_true',default=False)
parser.add_argument('--useGraphPooling',action='store_true',default=True)
parser.add_argument('--useNCE',action='store_true',default=False)


parser.add_argument('--softLabelReg', type=float,default=1e-1)
parser.add_argument('--nodeSimReg', type=float,default=1e-3)
parser.add_argument('--NodeFeatureReg', type=float,default=1e-1)
parser.add_argument('--ClusterMatchingReg', type=float,default=1e-1)
parser.add_argument('--RandomWalkConsistencyReg', type=float,default=1e-1)
parser.add_argument('--graphPoolingReg', type=float,default=1e-1)
parser.add_argument('--pathLength', type=int,default=8)
parser.add_argument('--MixUpReg', type=float,default=1e-1)
parser.add_argument('--NCEReg', type=float,default=1e-1)
parser.add_argument('--seed', type=int,default=1)
parser.add_argument('--trialId', type=int,default=1)

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
    device = get_free_gpu(5) # more than 10GB free GPU memory
else:
    device = None
if args['device_id']>=0:
    device = args['device_id']
print (colored(f"using device: {device}",'red','on_yellow'))


torch.manual_seed(args["seed"])


dataset_name = args["dataset"]



# args['num_classes'] = 1  # binary cls
args['teacher_hidden_dim'] = 300 #emb_dim
if args['use_AdditionalAttr'] and args['usePE']:
    args['node_dim'] = 9+12  # 9: atom feature; 12: laPE
else:
    args['node_dim'] = 9



model_saving_path = f"checkpoint_models/{dataset_name}{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['drop_ratio']}_num_layers_{args['num_layer']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_nodeSimReg_{args['nodeSimReg']}_KD_method_{args['KD_name']}/trialId_{args['trialId']}/"
args["model_saving_path"] = model_saving_path

if args['use_KD']:
    result_saving_path = f"KD_results/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['drop_ratio']}_num_layers_{args['num_layer']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_nodeSimReg_{args['nodeSimReg']}_KD_method_{args['KD_name']}/trialId_{args['trialId']}/"
    args["model_saving_path"] = None
    args["result_saving_path"] = result_saving_path

else:
    if args["studentModelName"] == "GIN":
        args["model_saving_path"] = model_saving_path
    args["result_saving_path"] = f"KD_results/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['drop_ratio']}_num_layers_{args['num_layer']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_noKD/trialId_{args['trialId']}/"


if args['teacherModelName'] == "GIN":
    if dataset_name == "ogbg-molhiv":
        modelname_list = get_pt_files("best_models/MOLHIV/",model_name=None)
        print(modelname_list)

    elif dataset_name == "ogbg-molpcba":
        modelname_list = get_pt_files("best_models/MOLPCBA/",model_name=None)

    dataset = PygGraphPropPredDataset(root='data/GA_MLP/',name = dataset_name)
    teacherModel = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = 5, emb_dim = 300, drop_ratio = 0.5, virtual_node = True,graph_pooling="attention")
    if device is not None:
        s = torch.load(modelname_list[0],map_location=torch.device(f"cuda:{device}"))
        teacherModel.load_state_dict(torch.load(modelname_list[0],map_location=torch.device(f"cuda:{device}")))
    else:
        teacherModel.load_state_dict(torch.load(modelname_list[0],map_location=torch.device(f"cpu")))
    teacherModel.to(device)
    teacherModel.eval()


if args['studentModelName'] == "MLP":
    print (colored(f"using MLP as student model",'red','on_yellow'))
    args['num_tasks'] = dataset.num_tasks
    student_model = MLP(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GA-MLP":
    print (colored(f"using GA-MLP as student model",'red','on_yellow'))
    args['num_tasks'] = dataset.num_tasks
    args['hidden_dim']=300
    student_model = GA_MLP(**args)
    args["model"] = student_model    

s= time()
print (colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

transforms = []

if args['useDropoutEdge']:
    transforms.append(DropEdge(p=0.07))
if args['useRandomWalkConsistency']:
    transforms.append(RandomPathTransform(path_length=args["pathLength"]))
if args['useMixUp']:
    args['teacherModel']=teacherModel


transforms.append(TeacherModelTransform(teacherModel,use_clustering=args["useClusterMatching"],cluster_algo=args["clusterAlgo"]))
transforms = Compose(transforms)


if not args['use_AdditionalAttr']:
    dataset = PygGraphPropPredDataset(root='data/GA_MLP/',name = dataset_name,transform=transforms)
else:
    if args['studentModelName'] == "GA_MLP":
        dataset = PygGraphPropPredDataset(root='data/GA_MLP/',name = dataset_name,transform=transforms)
    else:
        dataset = PygGraphPropPredDataset(root='data/GA_MLP/',name = dataset_name,transform=transforms)
split_idx = dataset.get_idx_split()

    # for test
    # train_dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='test',name=dataset_name)
    # val_dataset = train_dataset
    # test_dataset = train_dataset


t = time()
print (t-s, f"seconds used to load dataset {dataset_name}")


if args['studentModelName'] == "GA-MLP":
    workers = 0  # some implementations hinders multiple workers processing ofr GA-MLP
else:
    workers = 2
# for real-use
train_dloader = DataLoader(dataset[split_idx["train"]],batch_size=args['batch_size'],shuffle=True,num_workers=workers if torch.cuda.is_available() else 0)
valid_dloader = DataLoader(dataset[split_idx["valid"]],batch_size=128,shuffle=False, num_workers=workers if torch.cuda.is_available() else 0)
test_dloader = DataLoader(dataset[split_idx["test"]],batch_size=128,shuffle=False,num_workers=workers if torch.cuda.is_available() else 0)
args['test_loader'] = test_dloader
args['valid_loader'] = valid_dloader
args['eval_metric'] = dataset.eval_metric

evaluator = Evaluator(dataset_name)
args['evaluator'] = evaluator

print (colored(f"dataset: {next(iter(train_dloader))}",'red','on_white'))

# for test
# test_dloader = DataLoader(dataset[:60],batch_size=32,shuffle=False,num_workers=8 if torch.cuda.is_available() else 0)
# args['val_loader'] = test_dloader
# args['test_loader'] = test_dloader


checkpoint_callback = ModelCheckpoint(
    dirpath=model_saving_path,  # Directory to save checkpoints
    filename='{epoch}-{val_auc:.4f}',  # Checkpoint file name
    save_top_k=1,  # Save the top 3 models
    verbose=True,
    monitor='rocauc',  # Quantity to monitor for saving
    mode='max'  # 'min' for minimizing the monitored quantity, 'max' for maximizing
)

pl_model = PL_UniversalModel(**args)

trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/stu_{args["studentModelName"]}_teacher_{args["teacherModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),checkpoint_callback],enable_checkpointing=True)
trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=None)

torch.cuda.empty_cache()



