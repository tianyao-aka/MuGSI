from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import nn
from base_GNN_models import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import torch
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


class PL_MLP(pl.LightningModule):
    def __init__(self,num_layers, input_dim, hidden_dim, num_classes, dropout_rate=0.2,lr=5e-3,weight_decay=1e-6):
        super(PL_MLP, self).__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.lr=lr
        self.weight_decay=weight_decay
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
        self.val_acc=[]
        self.record_acc = []



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y,batch = batch.x,batch.y, batch.batch
        h = self(x)
        h=global_add_pool(h,batch=batch)
        y_hat=self.pred(h)

        if self.num_classes > 1:
            # Classification problem: use log softmax loss
            loss = F.nll_loss(F.log_softmax(y_hat, dim=1), y)
        else:
            # Regression problem: use mean squared error loss
            loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss


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


