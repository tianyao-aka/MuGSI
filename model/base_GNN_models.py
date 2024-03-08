from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINEConv,GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
import sys
sys.path.append('../')
# from nov.dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,remove_self_loops


def weight_reset(m):
    if isinstance(m, nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()



class GIN(torch.nn.Module):
    def __init__(self, num_layers=4, hidden_dim=32,num_classes=10,dropout=0.,pooling_method='attention', *args, **kargs):
        super(GIN, self).__init__()
        dataset_name = kargs['dataset_name']
        
        edge_dim=None
        dataset = kargs['pyg_dataset']
        if dataset_name == 'CIFAR10':
            self.num_features = 5
        elif dataset_name == 'PATTERN':
            self.num_features = 3
        else:
            self.num_features = dataset.num_features
        hidden = hidden_dim
        if 'edge_attr' in dataset[0] and 0:
            edge_dim = dataset[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim,hidden)
            self.conv1 = GINEConv(
                Sequential(
                    Linear(dataset.num_features, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU()
                ),
                train_eps=True,edge_dim=hidden)
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(self.num_features, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True)
        self.convs = torch.nn.ModuleList()
        self.dropout_val = dropout
        for i in range(num_layers - 1):
            if 'edge_attr' in dataset[0]:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True,edge_dim=hidden))
            else:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(hidden, num_classes)
        self.apply(weight_reset)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data,output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x[:,:self.num_features]
        if 'edge_attr' in data:
            e = data.edge_attr
            e = self.edge_emb(e)
            x = self.conv1(x, edge_index,e)
        else:
            x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            if 'edge_attr' in data:
                x = conv(x, edge_index,e)
            else:
                x = conv(x, edge_index)
            xs += [x]
        if not output_emb:
            h = F.relu(self.lin1(torch.cat(xs, dim=1)))
            x = self.pool(h,data.batch)
            if self.dropout_val>0:
                x = self.dropout(x)
            x = self.pred(x)
            return x
        else:
            h = F.relu(self.lin1(torch.cat(xs, dim=1)))
            x = self.pool(h,data.batch)
            if self.dropout_val>0:
                x = self.dropout(x)
            emb = self.pred(x)
            return emb,h,x # x is the graph embedding after pooling layer



