from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINEConv,GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
import sys
# from nov.dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from torch_geometric.datasets import TUDataset


class MLP(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_classes,pooling_method='sum',**kargs):
        super(MLP, self).__init__()
        self.node_dim = node_dim
        self.layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        self.pred_layer = nn.Linear(2*hidden_dim, num_classes)
        if pooling_method=='sum':
            self.pool = global_add_pool
        if pooling_method=='attention':
            # self.pool = AttentionalAggregation(gate_nn=nn.Linear(2*hidden_dim, 1))
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                                                     nn.ReLU(),
                                                                     nn.BatchNorm1d(hidden_dim),
                                                                     nn.Linear(hidden_dim, 1)))

    def forward(self,data,output_emb = False):
        x = data.x[:,:self.node_dim]
        batch = data.batch
        if not output_emb:
            x = self.layers(x)
            return self.pred_layer(self.pool(x,data.batch))
        else:
            # also output node emb for KD
            h = self.layers(x)
            g = self.pool(h,data.batch)
            return self.pred_layer(g),h,g # g is the pooled graph embeddings
        
    
    
class GA_MLP(nn.Module):
    def __init__(self, node_dim,edge_dim, hidden_dim, num_classes,use_edge_feats=True,pooling_method='sum',num_hops=1,**kargs):
        super(GA_MLP, self).__init__()
        self.node_dim = node_dim
        self.use_edge = use_edge_feats
        self.K = num_hops
        self.layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        self.hop1_layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        
        self.hop2_layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        if use_edge_feats:
            self.edge_layers = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                BN(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2*hidden_dim),
                BN(2*hidden_dim),
                nn.ReLU(),
                nn.Linear(2*hidden_dim, 2*hidden_dim),
                BN(2*hidden_dim),
                nn.ReLU())
            
        self.pred_layer = nn.Linear(2*hidden_dim, num_classes)
        if pooling_method=='sum':
            self.pool = global_add_pool
        if pooling_method=='attention':
            # self.pool = AttentionalAggregation(gate_nn=nn.Linear(2*hidden_dim, 1))
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                                                     nn.ReLU(),
                                                                     nn.BatchNorm1d(hidden_dim),
                                                                     nn.Linear(hidden_dim, 1)))

    def forward(self,data,output_emb = False,testSpeed=False):
        x = data.x[:,:self.node_dim]
        hop1 = data.hop1_feature[:,:self.node_dim]
        batch = data.batch
        if not output_emb:
            x = self.layers(x)
            h1 = self.hop1_layers(hop1)
            h = x+h1
            if self.use_edge and "edge_features" in data:
                e = self.edge_layers(data.edge_features)
                h = h + e
            if testSpeed:
                return self.pool(h,data.batch)
            return self.pred_layer(self.pool(h,data.batch))
        else:
            # also output node emb for KD
            x = self.layers(x)
            h1 = self.hop1_layers(hop1)
            h = x+h1
            if self.use_edge and "edge_features" in data:
                e = self.edge_layers(data.edge_features)
                h = h + e
            g = self.pool(h,data.batch)
            return self.pred_layer(g),h,g # g is the pooled graph embeddings
        
