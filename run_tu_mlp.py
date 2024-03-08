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
# # pytorch lightning
from torch_geometric.datasets import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from model.PL_MLP import PL_MLP
import argparse
import numpy as np
from model.data_utils import *
import sys
sys.path.append('model/')

from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime


use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None



def get_date_str():
    x = datetime.now()
    y = str(x.month)
    d = str(x.day)
    h = str(x.hour)
    return y+d+h

def run_one_fold(dataset,num_classes,input_dim,hidden_dim,dropout_rate=0.5,num_layers=2,lr=5e-3,weight_decay=1e-5,fold_index=0,max_epochs=300,**kargs):

    # in_dim = dataset.num_features + dataset[0].rw_feature.shape[1]   # in_dim is the concat of raw feature and rw feature
    model = PL_MLP(num_layers, input_dim, hidden_dim, num_classes, dropout_rate=dropout_rate,lr=lr,weight_decay=weight_decay)
    train_idx,test_idx = k_fold_without_validation(dataset,10)
    validation_acc = []
    #trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu' if not use_cuda else 'gpu', devices=1,
                         # enable_progress_bar=True)
    print(colored(f'running the {fold_index}-th fold','red','on_blue'))
    train_loader = DataLoader(dataset=dataset[train_idx[fold_index]],batch_size=64,shuffle=True)
    val_loader = DataLoader(dataset=dataset[test_idx[fold_index]],batch_size=64,shuffle=False)

    trainer = pl.Trainer(max_epochs=max_epochs,accelerator='cpu' if not use_cuda else 'gpu',devices=1,enable_progress_bar=True,logger=False)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    validation_acc.append(model.record_acc)
    print (colored(f"best valid acc for fold:{fold_index}:{torch.tensor(model.record_acc).view(-1,).max().item()}",'red','on_blue'))

    return model.record_acc


if __name__ =='__main__':
    # classification,dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn,
    # base_dropout=0.5,mhc_dropout=0.5,base_layer=2,
    # mhc_layer=1,mhc_num_hops=3,lr=5e-3,weight_decay=1e-5,is_tudataset=False,epochs=100

    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='which dataset to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='which dataset to use')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Base GNN dropout rate')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='MHC GNN layer numbers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='MHC GNN layer numbers')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=8e-3,
                        help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='max epochs')
    parser.add_argument('--result_dir', type=str, default='fill_in',
                        help='where to save results')
    parser.add_argument('--fold_index', type=int, default=1,
                        help='fold index')


if __name__ =='__main__':
    # x = load_dataset('MUTAG')
    # print (x[0])
    pl.seed_everything(12345)
    args = parser.parse_args()
    # dataset
    # pyg_dataset = load_dataset(args.dataset_name)
    dataset = TUDataset(root=f'data/tmp/{args.dataset}', name=args.dataset)
    dataset_name = args.dataset
    args = vars(args)
    args['num_classes']=dataset.num_classes
    args['input_dim'] = dataset.num_features
    args['dataset']=dataset
    print ('args:',args)
    validation_acc = []
    records = {}

    print (colored(f'only run {args["fold_index"]}-th fold','red'))
    now = get_date_str()
    metrics = run_one_fold(**args)
    # print (args)
    # print (args['base_gnn_str'])
    saved_path = f'result_mlp/separate_run/{dataset_name}/date_{now}/MLP_layers_{args["num_layers"]}_hidden_dims_{args["hidden_dim"]}_batch_size_{args["batch_size"]}_dropout_{args["dropout_rate"]}/fold_{args["fold_index"]}/'
    if not path.exists(saved_path):
        os.makedirs(saved_path,exist_ok=True)
    with open(saved_path+'results.pkl','wb') as f:
        pickle.dump(metrics,f)

    
