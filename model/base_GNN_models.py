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


class GCN(torch.nn.Module):
    def __init__(self, pyg_dataset, num_layers, hidden_dim,num_classes=2,dropout=0.,pooling_method='attention', *args, **kwargs):
        super(GCN, self).__init__()
        self.num_features = pyg_dataset.num_features
        self.conv1 = GCNConv(pyg_dataset.num_features, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin1 = torch.nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.pred = torch.nn.Linear(hidden_dim,num_classes)
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(num_layers*hidden_dim,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data,output_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x[:,:self.num_features]
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        if not output_emb:
            x = self.pool(torch.cat(xs, dim=1), batch)
            # x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            if self.dropout_val>0:
                x = self.dropout(x)
            x= self.pred(x)
            return x
        else:
            node_emb = torch.cat(xs, dim=1)
            graph_emb = self.pool(node_emb, batch)
            # x = global_mean_pool(x, batch)
            graph_emb = F.relu(self.lin1(graph_emb))
            if self.dropout_val>0:
                graph_emb = self.dropout(graph_emb)
            x= self.pred(graph_emb)
            return x,node_emb,graph_emb


    def __repr__(self):
        return self.__class__.__name__

class GCNEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels)
        self.edge_emb = nn.Linear(in_channels,out_channels,bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.edge_emb.reset_parameters()

    def forward(self, x,edge_index,edge_attr=None):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_feats = None
        if edge_attr is not None:
            edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0),edge_attr=edge_attr,fill_value=0.0)
            edge_feats = self.edge_emb(edge_attr)
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)


        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-5: Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm,edge_feats=edge_feats)
        if edge_attr is not None:
            return out,edge_attr
        else:
            return out

    def message(self, x_j, norm,edge_feats):
        # x_j has shape [E, out_channels]
        if edge_feats is not None:
        # Step 4: Normalize node features.
            return norm.view(-1, 1) * (x_j+edge_feats)
        else:
            return norm.view(-1, 1) * x_j

