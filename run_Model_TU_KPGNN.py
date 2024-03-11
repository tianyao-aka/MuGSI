import json
import math
import os
import sys
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('model/')
sys.path.append('..')
current_directory = os.path.dirname(os.path.abspath(__file__))
print (current_directory)
if current_directory not in sys.path:
    sys.path.append(current_directory)
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
from model.base_GNN_models import GIN,GCN
from model.PL_UniversalModel import PL_UniversalModel
from model.PL_UniversalModel_TU import PL_UniversalModel_TU
# from model.PL_UniversalModel_TU_time_memory import PL_UniversalModel_TU
from model.data_cache import DatasetVars
import argparse
import warnings
from glob import glob

from KPGNN.layers.input_encoder import LinearEncoder

from model.training_utils import edge_feature_transform,extract_multi_hop_neighbors,post_transform
import torch_geometric.transforms as T
from KPGNN.layers.layer_utils import make_gnn_layer
from KPGNN.models.GraphClassification import GraphClassification
from KPGNN.models.model_utils import make_GNN
from copy import deepcopy as c

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments

parser.add_argument('--dataset', type=str,default="BZR")
parser.add_argument('--dataset_index', type=int,default=0)

parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='Probability of zeroing an activation in dropout layers.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU. Scales automatically when \
                        multiple GPUs are available.')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--l2_wd', type=float, default=3e-4, help='L2 weight decay.')
parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                    help="The kernel used for K-hop computation")
parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the model")
parser.add_argument("--model_name", type=str, default="KPGIN",
                    choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Base GNN model")
parser.add_argument("--K", type=int, default=2, help="Number of hop to consider")
parser.add_argument("--max_pe_num", type=int, default=30,
                    help="Maximum number of path encoding. Must be equal to or greater than 1")
parser.add_argument("--max_edge_type", type=int, default=1,
                    help="Maximum number of type of edge to consider in peripheral edge information")
parser.add_argument("--max_edge_count", type=int, default=30,
                    help="Maximum count per edge type in peripheral edge information")
parser.add_argument("--max_hop_num", type=int, default=5,
                    help="Maximum number of hop to consider in peripheral configuration information")
parser.add_argument("--max_distance_count", type=int, default=50,
                    help="Maximum count per hop in peripheral configuration information")
parser.add_argument('--wo_peripheral_edge', action='store_true',
                    help='If true, remove peripheral edge information from model')
parser.add_argument('--wo_peripheral_configuration', action='store_true',
                    help='If true, remove peripheral node configuration information from model')
parser.add_argument('--wo_path_encoding', action='store_true',
                    help='If true, remove path encoding information from model')
parser.add_argument('--wo_edge_feature', action='store_true',
                    help='If true, remove edge feature from model')
parser.add_argument("--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1")
parser.add_argument("--num_layer", type=int, default=2, help="Number of layer for feature encoder")
parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last", "concat"),
                    help="Jumping knowledge method")
parser.add_argument("--residual", action="store_true", help="If true, use residual connection between each layer")
parser.add_argument("--use_rd", action="store_true", help="If true, add resistance distance feature to model")
parser.add_argument("--virtual_node", action="store_true",
                    help="If true, add virtual node information in each layer")
parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN")
parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon in GIN model is trainable")
parser.add_argument("--combine", type=str, default="geometric", choices=("attention", "geometric"),
                    help="Combine method in k-hop aggregation")
parser.add_argument("--pooling_method", type=str, default="attention", choices=("mean", "sum", "attention"),
                    help="Pooling method in graph classification")
parser.add_argument('--norm_type', type=str, default="Batch",
                    choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                    help="Normalization method in model")
parser.add_argument('--aggr', type=str, default="add",
                    help='Aggregation method in GNN layer, only works in GraphSAGE')
parser.add_argument('--factor', type=float, default=0.5, help='Factor for reducing learning rate scheduler')
parser.add_argument('--reprocess', action="store_true", help='If true, reprocess the dataset')


parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=350)
parser.add_argument('--numWorkers', type=int,default=4)


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
args_copy = c(args)
if args.wo_path_encoding:
    args.num_hopk_edge = 1
else:
    args.num_hopk_edge = args.max_pe_num


def multihop_transform(g):
    return extract_multi_hop_neighbors(g, args.K, args.max_pe_num, args.max_hop_num, args.max_edge_type,
                                        args.max_edge_count,
                                        args.max_distance_count, args.kernel)

if args.use_rd:
    rd_feature = 0
else:
    def rd_feature(g):
        return g

transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

dataset = TUDataset(root=f'data/KPGNN/raw/hops_{args.K}/',name=args.dataset,pre_transform=T.Compose([edge_feature_transform, multihop_transform, rd_feature]),transform=transform)
num_features = dataset.num_features
num_classes=2
print (colored(dataset[0], 'red'))
print (colored(f"num_features:{num_features}", 'red'))
def get_model(args,input_size):
    layer = make_gnn_layer(args)
    init_emb = LinearEncoder(input_size, args.hidden_size,num_features=num_features)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob)

    model = GraphClassification(embedding_model=gnn,
                                pooling_method=args.pooling_method,
                                output_size=2)

    model.reset_parameters()
    return model

kpgnn= get_model(args,input_size=num_features)  
args = vars(args)
args['hidden_dim'] = args['hidden_size']
args['num_hops'] = args['K']
args['dropout'] = args['drop_prob']



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
    device = get_free_gpu(6) # more than 10GB free GPU memory
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
print (t)
args['num_classes'] = t['num_classes']
if args["usePE"]:
    args["node_dim"] = t["num_features"]+20
else:
    args["node_dim"] = t["num_features"]


model_saving_path = f"best_models/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_size']}_dropout_{args['drop_prob']}_num_layers_{args['num_layer']}_hops_{args['K']}_batch_size_{args['batch_size']}_kernel_{args['kernel']}_combine_{args['combine']}/fold_index_{args['dataset_index']}/"
loss_saving_path = f"LossCurves/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_size']}_dropout_{args['drop_prob']}_num_layers_{args['num_layer']}_hops_{args['K']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_kernel_{args['kernel']}_combine_{args['combine']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["loss_saving_path"] = loss_saving_path

if args['use_KD']:
    result_saving_path = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_size']}_dropout_{args['drop_prob']}_num_layers_{args['num_layer']}_hops_{args['K']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_kernel_{args['kernel']}_combine_{args['combine']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
    args["model_saving_path"] = None
    args["result_saving_path"] = result_saving_path
else:
    if "KP" in args["studentModelName"]:
        args["model_saving_path"] = model_saving_path
    else:
        args["model_saving_path"] = None
    args["result_saving_path"] = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_size']}_dropout_{args['drop_prob']}_num_layers_{args['num_layer']}_hops_{args['K']}_kernel_{args['kernel']}_combine_{args['combine']}_batch_size_{args['batch_size']}_usePE_{args['usePE']}_noKD/fold_index_{args['dataset_index']}/"


if args['studentModelName'] == "MLP":
    print (colored(f"using MLP as student model",'red','on_yellow'))
    student_model = MLP(**args)
    args["model"] = student_model

    
elif "KP" in args['studentModelName']:
    print (colored(f"using KPGNN as student model",'red','on_yellow'))
    args["model"] = kpgnn
    

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
    # raw_dataset = TUDataset(root=f"data/raw/",name=dataset_name)
    t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
    hidden,dropout,num_layers,num_classes = t["hidden"],t["dropout"],t["num_layers"],t["num_classes"]
    args["teacher_hidden_dim"] = hidden
    teacherModel = kpgnn
    TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
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

    transform = [post_transform(args_copy.wo_path_encoding, args_copy.wo_edge_feature)]
    if args['useDropoutEdge']:
        transform.append(DropEdge(p=0.07))
    if args['useRandomWalkConsistency']:
        transform.append(RandomPathTransform(path_length=args["pathLength"]))
    if args['useMixUp']:
        args['teacherModel']=teacherModel
    

    transform.append(TeacherModelTransform(teacherModel,use_clustering=args["useClusterMatching"],cluster_algo=args["clusterAlgo"]))
    transforms = Compose(transform)
    
    
s= time()
print (colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

dataset_idx = args['dataset_index']
dataset_path = f'data/KPGNN/withAdditionalAttr/hops_{args_copy.K}/'
if args["use_KD"]:
    tu_dataset = TUDataset(root=dataset_path,name=dataset_name,transform=transforms)
else:
    tu_dataset = TUDataset(root=f'data/KPGNN/raw/hops_{args_copy.K}/',name=dataset_name)
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
   patience=65,
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


