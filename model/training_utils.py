import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric import transforms as T
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from time import time
from torch_geometric.transforms import BaseTransform,Compose
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.transforms import BaseTransform
from scipy.sparse import coo_matrix
from torch_geometric.utils import is_undirected, to_undirected,contains_isolated_nodes
from torch_geometric.datasets import GNNBenchmarkDataset,TUDataset
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from pytorch_lightning.callbacks import Callback
from termcolor import colored
# from KPGNN_data_utils import get_peripheral_attr,adj_K_order
from torch_geometric.utils import to_scipy_sparse_matrix
from copy import deepcopy as c

def computeNCE(a, b, B):
    # a: context, in this case is student embedding. b: teacher, in this case is teacher embedding.
    # B: sample size
    T=2.0
    N = a.shape[0]
    indices = torch.multinomial(torch.ones(N), B, replacement=False)  # Randomly select B indices without replacement
    a_selected = a[indices]  # Get selected a vectors
    b_selected = b[indices]  # Get selected b vectors
    

    numerators = torch.sum(a_selected * b_selected, dim=1)
    denominator = torch.logsumexp(a_selected @ b.t(),dim=1)

    result = numerators - denominator
    return result.mean()




class NaNStopping(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss']
        if torch.isnan(loss):
            print("Loss is NaN, stopping training")
            trainer.should_stop = True

def fast_cosine_sim_matrix(a, b):
    # normalize each tensor (vector)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    # calculate cosine similarity using dot product
    cos_sim_matrix = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    return cos_sim_matrix


def calculate_conditional_probabilities(index_arr, feature_matrix):
    # Extract the first node for each row
    first_node_indices = index_arr[:, 0]
    first_node_features = feature_matrix[first_node_indices]
    
    # Reshape first_node_features for batch matrix multiplication
    first_node_features = first_node_features.unsqueeze(1)
    # Extract the remaining nodes for each row
    remaining_node_indices = index_arr[:, 1:]
    remaining_node_features = feature_matrix[remaining_node_indices].transpose(2,1)
    # Perform batch matrix multiplication
    dot_product = torch.bmm(first_node_features, remaining_node_features)
    # Squeeze the output to remove the extra dimension
    dot_product = dot_product.squeeze(1)
    return F.softmax(dot_product,dim=-1)

def nodeFeatureAlignment(stuNode, teacherNode):
    # Ensure that the input tensors are float tensors
    assert stuNode.shape[1]==teacherNode.shape[1], "Input tensors must have the same number of features along the last dimension in nodeFeatureAlignment method"
    
    stuNode = stuNode.float()
    teacherNode = teacherNode.float()

    # Compute the L2 norms of the input tensors along the last dimension
    stuNode_norm = torch.norm(stuNode, dim=-1, keepdim=True)
    teacherNode_norm = torch.norm(teacherNode, dim=-1, keepdim=True)

    # Normalize the input tensors
    stuNode_normalized = torch.nan_to_num(stuNode / stuNode_norm, nan=0.0)
    teacherNode_normalized = torch.nan_to_num(teacherNode / teacherNode_norm, nan=0.0)

    # Compute the squared L2 norm of the difference between the normalized tensors
    loss = F.mse_loss(stuNode_normalized, teacherNode_normalized)
    return loss


# generate random walk paths for path consistency regularization for KD. 
class RandomPathTransform(BaseTransform):
    def __init__(self, sample_size=20, path_length=15):
        super(RandomPathTransform, self).__init__()
        self.sample_size = sample_size
        self.path_length = path_length

    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None)
        try:
            random_paths = nx.generate_random_paths(G, self.sample_size, self.path_length)
            # print ("have isolated nodes:",contains_isolated_nodes(data.edge_index))
            data.random_walk_paths = torch.tensor(list(random_paths)).long()
            # print (1,data.random_walk_paths.shape)
            return data
        except:
            data.random_walk_paths = torch.ones((self.sample_size, self.path_length+1), dtype=torch.long)
            # print (2,data.random_walk_paths.shape)
            return data


    def __repr__(self):
        return '{}(sample_size={}, path_length={})'.format(self.__class__.__name__, self.sample_size, self.path_length)


def calculate_kl_loss(pred, target):
    # Calculate the KL divergence for each row
    kl_val = F.kl_div(input=torch.log(pred), target=target, reduction='batchmean')
    # Replace NaN or Inf with zero
    if torch.isnan(kl_val) or torch.isinf(kl_val):
        kl_val = 0.0
    return kl_val



def calc_KL_divergence(pred_logits, y,criterion=nn.KLDivLoss(reduction="batchmean")):
    preds = F.log_softmax(pred_logits, dim=1)
    y = F.softmax(y, dim=1)
    val = criterion(preds, y)
    if torch.isnan(val) or torch.isinf(val):
        val = 0.0
    return val

def calc_node_similarity(node_emb_stu,node_emb_teacher):
    X = node_emb_stu@node_emb_stu.t()
    Y = node_emb_teacher@node_emb_teacher.t()
    diff = X-Y
    norm = torch.norm(diff,p='fro')
    return norm


 
 
class TeacherModelTransform(BaseTransform):
    def __init__(self, model,use_clustering=False,cluster_algo = "louvain"):
        self.model = model
        self.use_clustering = use_clustering
        self.cluster_algo = cluster_algo
        self.model.eval()

    def __call__(self, data):
        self.model.eval()
        with torch.no_grad():
            pred,node_emb,graphEmb = self.model(data,output_emb=True)
            data.teacherPred = pred
            data.nodeEmb = node_emb
            data.graphEmb = graphEmb
            if self.use_clustering:
                if self.cluster_algo == "louvain":
                    cluster_id = data.louvain_cluster_id
                if self.cluster_algo == "metis5":
                    cluster_id = data.metis_clusters5
                if self.cluster_algo == "metis10":
                    cluster_id = data.metis_clusters10
                # if cluster_id.max()<=2:
                #     print (colored(f"cluster num is: {cluster_id}",'red'))
                h = self.model.pool(node_emb,cluster_id.view(-1,))
                data.teacherClusterInfo = h
        return data


def edge_feature_transform(data):
    if data.edge_attr is not None:
        data.edge_attr = torch.where(data.edge_attr == 1)[1] + 2
    return data


def post_transform(wo_path_encoding, wo_edge_feature):
    """Post transformation of dataset for KP-GNN
    Args:
        wo_path_encoding (bool): If true, remove path encoding from model
        wo_edge_feature (bool): If true, remove edge feature from model
    """
    if wo_path_encoding and wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            edge_attr[edge_attr > 2] = 2
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    elif wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 0]
            t[t > 2] = 2
            edge_attr[:, 0] = t
            g.edge_attr = edge_attr
            return g

    elif wo_path_encoding:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 1:]
            t[t > 2] = 2
            edge_attr[:, 1:] = t
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    else:
        def transform(g):
            return g

    return transform


def extract_multi_hop_neighbors(data, K, max_edge_attr_num, max_hop_num,
                                max_edge_type, max_edge_count, max_distance_count, kernel):
    """generate multi-hop neighbors for input PyG graph using shortest path distance kernel
    Args:
        data (torch_geometric.data.Data): PyG graph data instance
        K (int): number of hop
        max_edge_attr_num (int): maximum number of encoding used for hopk edge
        max_hop_num (int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type (int): maximum number of edge type to consider
        max_edge_count (int): maximum number of count for each type of edge
        max_distance_count (int): maximum number of count for each distance
        kernel (str): kernel used to extract neighbors
    """
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    # graph with no edge
    if edge_index.size(1) == 0:
        edge_matrix_size = [num_nodes, K, max_edge_type, 2]
        configuration_matrix_size = [num_nodes, K, max_hop_num]
        peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
        peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
        data.peripheral_edge_attr = peripheral_edge_matrix
        data.peripheral_configuration = peripheral_configuration_matrix
        return data

    if "edge_attr" in data:
        edge_attr = data.edge_attr
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr = (torch.ones([edge_index.size(-1)]) * 2).long()  # E

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes).toarray()).long()
    # compute each order of adj
    adj_list = adj_K_order(adj, K)

    if kernel == "gd":
        # create K-hop edge with graph diffusion kernel
        final_adj = 0
        for adj_ in adj_list:
            final_adj += adj_
        final_adj[final_adj > 1] = 1
    else:
        # process adj list to generate shortest path distance matrix with path number
        exist_adj = c(adj_list[0])
        for i in range(1, len(adj_list)):
            adj_ = c(adj_list[i])
            # mask all the edge that already exist in previous hops
            adj_[exist_adj > 0] = 0
            exist_adj = exist_adj + adj_
            exist_adj[exist_adj > 1] = 1
            adj_list[i] = adj_
        # create K-hop edge with sortest path distance kernel
        final_adj = exist_adj

    g = nx.from_numpy_array(final_adj.numpy(), create_using=nx.DiGraph)
    edge_list = g.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    hop1_edge_attr = edge_attr_adj[edge_index[0, :], edge_index[1, :]]
    edge_attr_list = [hop1_edge_attr.unsqueeze(-1)]
    pe_attr_list = []
    for i in range(1, len(adj_list)):
        adj_ = c(adj_list[i])
        adj_[adj_ > max_edge_attr_num] = max_edge_attr_num
        # skip 1 as it is the self-loop defined in the model
        adj_[adj_ > 0] = adj_[adj_ > 0] + 1
        adj_ = adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
        pe_attr_list.append(torch.diag(adj_).unsqueeze(-1))
    edge_attr = torch.cat(edge_attr_list, dim=-1)  # E * K
    if K > 1:
        pe_attr = torch.cat(pe_attr_list, dim=-1)  # N * K-1
    else:
        pe_attr = None

    peripheral_edge_attr, peripheral_configuration_attr = get_peripheral_attr(adj_list, edge_attr_adj, max_hop_num,
                                                                              max_edge_type, max_edge_count,
                                                                              max_distance_count)
    # update all the attributes
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.peripheral_edge_attr = peripheral_edge_attr
    data.peripheral_configuration_attr = peripheral_configuration_attr
    data.pe_attr = pe_attr
    return data


def edge_feature_transform(data):
    if data.edge_attr is not None:
        data.edge_attr = torch.where(data.edge_attr == 1)[1] + 2
    return data



if __name__ == '__main__':
    from termcolor import colored
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(current_dir, 'data/')
    # dataset = TUDataset(root=f'data/PROTEINS/',name="PROTEINS")
    
    # load model
    
    # modelname_list = get_pt_files("best_models/CIFAR10/")
    # dataset = GNNBenchmarkDataset(root=f"data_raw/CIFAR10/",split='test',name="CIFAR10")
    # model = GIN(num_classes=10,pyg_dataset=dataset)
    # model.load_state_dict(torch.load(modelname_list[0],map_location=torch.device('cpu')))
    # print (model)
    # dataset = GNNBenchmarkDataset(root=f"data_raw/CIFAR10",split='test',name="CIFAR10",transform=TeacherModelTransform(model))
    # # print (dataset)
    # print (dataset[0])
    # init two rand tensor
    x = torch.randn((100,10))
    y = torch.randn((100,10))
    print (calc_node_similarity(x,y))
    
    