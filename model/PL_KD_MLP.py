import torch
from torch_geometric.nn import global_mean_pool, global_add_pool
from base_GNN_models import *
import numpy as np
import pandas as pd
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class PL_KD_MLP(pl.LightningModule):
    def __init__(self,num_layers, input_dim, hidden_dim, num_classes, dropout_rate=0.5,lr=5e-3,weight_decay=1e-6,gnn_model=None,use_node_sim=False,lamda1=1e-1,lamda2=1e-6):
        super(PL_KD_MLP, self).__init__()
        self.save_hyperparameters(ignore=['gnn_model'])
        self.use_node_sim = use_node_sim
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.lr=lr
        self.weight_decay=weight_decay
        self.lambda1=lamda1
        self.lambda2=lamda2
        self.acc = Accuracy(top_k=1)
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        if dropout_rate > 0:
            self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_layers - 2):  # -2 because we add the input and output layers separately
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.pred = nn.Linear(hidden_dim, num_classes if num_classes > 1 else 1)
        self.proj = nn.Linear(hidden_dim,gnn_model.hidden_dim)


        # pretrained model
        self.gnn_model = gnn_model
        self.gnn_model.eval()
        self.val_acc=[]
        self.record_acc = []

    def compute_node_similarity(self,X,batch):
        num_graphs = batch.num_graphs
        node_similarity_list = []

        for i in range(num_graphs):
            # Get the nodes for the current graph
            node_mask = batch.batch == i
            X_i = X[node_mask]

            # Compute node similarity for the current graph
            # Assuming the features are already normalized
            node_similarity = torch.matmul(X_i, X_i.t())
            node_similarity_list.append(node_similarity)

        return node_similarity_list

    def calc_squared_frobenius_norm(self,A, B):
        diff = A - B
        return torch.sum(diff * diff)

    def cls_output_from_GNN(self,batch):
        with torch.no_grad():
            node_emb,soft_label = self.gnn_model(batch,test_stage=True)
        return node_emb,soft_label


    def categorical_CE(self,pred, groud_truth):
        return -(groud_truth * torch.log(pred)).sum(dim=1).mean()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y,batch_idx = batch.x,batch.y, batch.batch
        h = self(x)
        h_sum=global_add_pool(h,batch=batch_idx)
        y_hat=self.pred(h_sum)
        y_hat_prob=F.softmax(y_hat,dim=-1)
        # use soft labels from GNN
        node_emb, soft_label = self.cls_output_from_GNN(batch)
        soft_loss = self.categorical_CE(y_hat_prob,soft_label)
        if self.use_node_sim:
            mlp_emb = F.relu(self.proj(h))
            mlp_node_sim_list = self.compute_node_similarity(mlp_emb,batch)
            gnn_node_sim_list = self.compute_node_similarity(node_emb,batch)
            N = len(mlp_node_sim_list)
            assert len(mlp_node_sim_list)==len(gnn_node_sim_list), "node sim list from MLP and GNN should be the same"
            rsd_loss = torch.tensor([self.calc_squared_frobenius_norm(mlp_node_sim_list[i],gnn_node_sim_list[i]) for i in range(N)])
            rsd_loss = torch.mean(rsd_loss)

        if self.num_classes > 1:
            # Classification problem: use log softmax loss
            loss = F.nll_loss(F.log_softmax(y_hat, dim=1), y)
        else:
            # Regression problem: use mean squared error loss
            loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss,prog_bar=True,on_epoch=True)
        self.log('soft_loss', soft_loss, prog_bar=True, on_epoch=True)
        if self.use_node_sim:
            self.log('rsd_loss', rsd_loss, prog_bar=True, on_epoch=True)
        if self.use_node_sim:
            return loss + self.lambda1*soft_loss + self.lambda2*rsd_loss
        else:
            return loss + self.lambda1 * soft_loss


    def validation_step(self, batch, batch_idx):
        x, y, batch = batch.x, batch.y, batch.batch
        h = self(x)
        h=global_add_pool(h,batch=batch)
        h=self.pred(h)
        if self.num_classes>1:
            y = y.view(-1,)
            loss_val = F.nll_loss(h, y)
            acc = self.acc(h,y)
            metrics = {'val_acc':acc}
            pred = torch.argmax(h, dim=1)
            tot = len(y)
            correct = torch.sum(y == pred).item()
            self.val_acc.append((tot, correct))
        else:
            loss_val = self.l1_loss(h,y)
            metrics = { 'val_loss': loss_val}
        self.log_dict(metrics,prog_bar=True,logger=True,on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        tot = sum([i[0] for i in self.val_acc])
        correct = sum([i[1] for i in self.val_acc])
        val = 1.*correct/tot
        self.record_acc.append(val)
        self.val_acc = []
        self.log('val_acc_per_epoch',val,prog_bar=True)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
        return {'optimizer':optimizer,'lr_scheduler':scheduler}


if __name__ =='__main__':
    dataset = TUDataset(root='data/tmp/mutag',name='MUTAG')
    print (dataset[0])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    tmp = next(iter(train_loader))
    model = PL_MLP(3, dataset.num_features, 64, dataset.num_classes)
    out = model(tmp.x)
    print (out.shape)


