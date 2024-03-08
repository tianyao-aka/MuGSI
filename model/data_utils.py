import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric import transforms as T
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from time import time
from torch_geometric.transforms import BaseTransform,Compose,AddLaplacianEigenvectorPE
from torch_geometric.utils import degree
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
import community as community_louvain
from torch_geometric.transforms import BaseTransform
import random
import pymetis
from scipy.sparse import coo_matrix
from tqdm import tqdm
import os
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.datasets import GNNBenchmarkDataset,TUDataset
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import inv
from pynvml import *
import shutil
import glob
from datetime import datetime
from termcolor import colored
import scipy.sparse as sp
from scipy.sparse import diags
import numpy as np



def get_pt_files(root_dir,model_name="GIN"):
    # Find all directories under root_dir containing "GIN" in their names
    if model_name is not None:
        gin_dirs = [dirpath for dirpath, dirnames, filenames in os.walk(root_dir) if model_name in dirpath]
    else:
        gin_dirs = [dirpath for dirpath, dirnames, filenames in os.walk(root_dir)]

    # Initialize an empty list to store all .pt files
    pt_files = []

    # Iterate through each directory containing "GIN"
    for gin_dir in gin_dirs:
        # Find all .pt files in the directory
        pt_files_in_dir = glob.glob(os.path.join(gin_dir, '*.pt'))
        
        # Append the found .pt files to the master list
        pt_files.extend(pt_files_in_dir)
    pt_files.sort()
    return pt_files # make sure the files are returned in the same order every time


def get_free_gpu(thres): # in GB
    # Initialize NVML
    gpu_list = []
    try:
        nvmlInit()
    except Exception as e:
        return -1
    # Get number of available GPU devices
    device_count = nvmlDeviceGetCount()

    for i in range(device_count):
        # Get handle for the GPU device
        handle = nvmlDeviceGetHandleByIndex(i)
        
        # Get memory info for the GPU device
        mem_info = nvmlDeviceGetMemoryInfo(handle)

        # Check if the free memory is greater than 20 GB
        if mem_info.free / (1024 ** 3) > thres:  # convert bytes to gigabytes
            # Return the device ID
            gpu_list.append((mem_info.free / (1024 ** 3),i))
    if len(gpu_list)==0: 
        # If no GPU with sufficient memory was found, return None
        return None
    else:
        gpu_list = sorted(gpu_list,key=lambda x:x[0],reverse=True)
        print (colored(f'using device {gpu_list[0][1]}, free GPU memory is: {gpu_list[0][0]}','red','on_white'))
        return gpu_list[0][1]



def write_results_to_file(fpath, n, s):
    # Check if the directory exists, if not, create it
    # fpath: 存放路径
    # n: 文件名 （.txt结尾）
    # s: 内容
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            pass

    # Construct full file path
    full_path = os.path.join(fpath, n)

    # Open the file in write mode, which will create the file if it does not exist
    # and overwrite it if it does. Then write the string to the file.
    with open(full_path, 'w') as f:
        f.write(s)



def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    else:
        pass

def read_results(directory):
    files_content = []
    for filename in glob.iglob(directory + '**/*.txt', recursive=True):
        with open(filename, 'r') as file:
            files_content.append(file.read())
    return files_content


def save_model_weights(model, path,file_name):
    """
    Saves the model weights to the provided path.

    Args:
    model : PyTorch model
    path : Directory to save the model weights

    """
    # Check if the directory exists
    if not os.path.exists(path):
        # If not, create the directory
        os.makedirs(path)
    else:
        # If directory exists, delete all files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    # Save the model weights
    torch.save(model.state_dict(), os.path.join(path, file_name))


# for TU dataset
def degree_post_processing(data):
    deg_func = T.OneHotDegree(max_degree=10000)
    data = deg_func(data)
    num_feats = data.nfeats
    N = data.x.shape[0]
    degrees = degree(data.edge_index[0],num_nodes=N).view(-1,1).float()
    max_degree = degrees.max().item()
    degrees = degrees/max_degree
    # print ('num feats:',num_feats)
    if num_feats>0:
        feature = data.x[:,:num_feats]
        deg_feats = data.x[:,num_feats:]
        val = torch.cat([degrees,deg_feats],dim=1)
        val = val[:,:65]
        data.x = torch.cat((feature,val),dim=1)
    else:
        val = torch.cat([degrees,data.x],dim=1)
        val = val[:,:65]
        data.x = val
    return data

class DegreeTransform(object):
    # combine position and intensity feature, ignore edge value
    def __init__(self) -> None:
        self.deg_func = T.OneHotDegree(max_degree=10000)

    def __call__(self, data):
        data = self.deg_func(data)
        N = data.x.shape[0]
        degrees = degree(data.edge_index[0],num_nodes=N).view(-1,1).float()
        max_degree = degrees.max().item()
        degrees = degrees/max_degree
        val = torch.cat([degrees,data.x],dim=1)
        val = val[:,:65]
        data.x = val
        return data


# for CIFAR10
class SuperpixelTransform(object):
    # combine position and intensity feature, ignore edge value
    def __call__(self, data):
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data.edge_attr = None # remove edge_attr
        data.edge_index = to_undirected(data.edge_index) 
        return data



class DropEdge(BaseTransform):
    def __init__(self, p=0.5):
        """
        Initialize the transform. 
        p: The probability of an edge being dropped.
        """
        self.p = p

    def __call__(self, data):
        """
        Apply the transform. 
        data: A Data object, which includes edge_index attribute.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Create a mask of edges to keep
        mask = torch.rand(num_edges) > self.p
        data.edge_index = to_undirected(edge_index[:, mask])

        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)



def get_cluster_info_louvain(X, cluster_id, max_cluster_num, ptr):
    # X: should be hidden feature matrix from model output for all graphs
    # cluster_id: cluster id for each graph
    # max_cluster_num: cluster members for each graph, is a list of lists.
    
    num_graphs = len(max_cluster_num)
    # ptr = ptr + [len(cluster_id)]  # Add total number of nodes for easier slicing

    graph_cluster_features = []
    cluster_id=cluster_id.view(-1,)
    for i in range(num_graphs):
        graph_features = []

        # Get cluster ids for the current graph
        graph_cluster_ids = cluster_id[ptr[i]:ptr[i+1]]

        for cluster_idx in range(max_cluster_num[i]):
            # Get indices for nodes in the current cluster
            idx = (graph_cluster_ids == cluster_idx).nonzero(as_tuple=True)[0]

            # Add the feature matrix for the current cluster to the list
            graph_features.append(X[ptr[i]:ptr[i+1]][idx])

        # Add the list of feature matrices for the current graph to the result
        graph_cluster_features.append(graph_features)

    return graph_cluster_features # return a list of list. Each inner list represents a graph, the elements in the list represents clusters.



def get_cluster_info(X,cluster_num,cluster_member,ptr):
    # X: should be hidden feature matrix from model output for all graphs
    # cluster_num: number of clusters in a single graph
    # cluster_member: cluster members for each graph, is a list of lists.
    # ptr: bias to the node indices
    
    cluster_features = []

    # Add bias to cluster members
    if ptr>0:
        cluster_member = [[member + ptr for member in members] for members in cluster_member]

    # Retrieve feature matrix for each cluster
    for members in cluster_member:
        if len(members) >= 3:  # Skip clusters with less than 3 nodes
            cluster_features.append(X[members])

    return cluster_features, len(cluster_features)

def random_walk(data, l):
    # Add self-loops to the adjacency matrix
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # Create adjacency matrix
    row, col = edge_index
    A = coo_matrix((torch.ones(row.size(0)), (row.numpy(), col.numpy())), shape=(data.num_nodes, data.num_nodes))

    # Create degree matrix
    deg = A.sum(1)
    D = coo_matrix((deg.A1, (range(data.num_nodes), range(data.num_nodes))), shape=(data.num_nodes, data.num_nodes))

    # Compute inverse of degree matrix
    D_inv = D.power(-1).todense()

    # Compute normalized adjacency matrix
    A_hat = D_inv @ A

    # Initialize H with identity matrix
    H = torch.eye(data.num_nodes)

    # Perform random walk for l steps
    res = [H.diag().unsqueeze(1)]
    for _ in range(l):
        H = torch.from_numpy(A_hat @ H.detach().numpy()).float()
        res.append(H.diag().unsqueeze(1))

    # Concatenate results and transpose to get a tensor of shape [N, l]
    return torch.cat(res, dim=1)[:,1:]



def k_fold_without_validation(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for tr_idx, test_idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        train_indices.append(torch.from_numpy(tr_idx))
        test_indices.append(torch.from_numpy(test_idx))
    return train_indices, test_indices


class ComputeKhopNeighbors(BaseTransform):
    # pretransform a dataset to have attr: Khop_neighbors
    # Khop_neighbors hold a list of |V| lists, each list have K lists, containing the nodes from 1 up to K hop.
    def __init__(self, K):
        super(ComputeKhopNeighbors, self).__init__()
        self.K = K

    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None,to_undirected=True)  # Convert to NetworkX graph

        khop_neighbors = [[] for _ in range(data.num_nodes)]

        for node in G.nodes:
            path_lengths = nx.single_source_shortest_path_length(G, node)

            grouped_nodes = {}
            for target, length in path_lengths.items():
                if length <= self.K:
                    if length not in grouped_nodes:
                        grouped_nodes[length] = []
                    grouped_nodes[length].append(target)

            for k in range(1, self.K + 1):
                khop_neighbors[node].append(grouped_nodes.get(k, []))

        data.Khop_neighbors = khop_neighbors

        return data

class ComputeClusteringCoefficient(BaseTransform):
    # compute clustering coefficient

    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None,to_undirected=True)  # Convert to NetworkX graph

        # Compute clustering coefficients
        clustering_coeffs = nx.clustering(G)

        # Convert the clustering coefficients into a tensor and add as an attribute to data
        cc_values = torch.tensor([clustering_coeffs[node] for node in G.nodes()]).view(-1,1)
        data.CC_value = cc_values

        return data


def metis_cluster_to_tensor(lst):
    # Calculate the total number of nodes
    num_nodes = sum(len(sublist) for sublist in lst)

    # Initialize tensor with all values as -1
    tensor = torch.full((num_nodes, 1), -1)

    # Loop through the list of lists
    for cluster_id, sublist in enumerate(lst):
        for node in sublist:
            # Assign each node to its cluster id
            tensor[node] = cluster_id

    return tensor

class PerformMetisClustering(BaseTransform):
    def __init__(self,n_clusters) -> None:
        super(PerformMetisClustering,self).__init__()
        self.n_clusters = n_clusters
    
    def __call__(self, data):
        n = data.num_nodes  # number of nodes
        adjacency_list = [np.array([]) for _ in range(n)]

        for i in range(data.edge_index.shape[1]):
            src, dest = data.edge_index[:, i].numpy()
            adjacency_list[src] = np.append(adjacency_list[src], dest)
            adjacency_list[dest] = np.append(adjacency_list[dest], src)

        # Remove duplicates (since this is an undirected graph)
        adjacency_list = [np.unique(adj_list) for adj_list in adjacency_list]

        # Now you have adjacency list in the required format.
        # Use PyMetis to partition the graph

        val, membership = pymetis.part_graph(self.n_clusters, adjacency=adjacency_list)
        clusters = [[] for _ in range(self.n_clusters)]
        for node, cluster in enumerate(membership):
            clusters[cluster].append(node)
        clusters = metis_cluster_to_tensor(clusters)
        # clusters = torch.tensor(clusters).view(-1,1)
        if self.n_clusters==5:
            data.metis_clusters5 = clusters
        if self.n_clusters==10:
            data.metis_clusters10 = clusters
        if self.n_clusters==15:
            data.metis_clusters15 = clusters
        if self.n_clusters==20:
            data.metis_clusters20 = clusters
        return data


class PerformLouvainClustering(BaseTransform):
    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True)

        # Perform Louvain clustering
        partition = community_louvain.best_partition(G)
        # Convert partition (a dict) to a list of cluster labels
        labels = list(partition.values())
        data.louvain_cluster_id = torch.tensor(labels).view(-1, 1)
        return data

class AddAllOnes(BaseTransform):
    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        data.x = torch.ones((N,1)).to(device)
        return data


class KHopTransform(BaseTransform):
    def __init__(self, K=[1, 2],agg='sum'):
        super(KHopTransform, self).__init__()
        self.K = K
        self.agg=agg

    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        edge_index = data.edge_index
        
        hop_features = []
        for k in self.K:
            # Create adjacency matrix A and raise it to the power of k
            A_k = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), size=(N, N))
            A_k = torch.sparse.mm(A_k, A_k) if k == 2 else A_k  # if k > 2, need to multiply more times
            
            D_k = A_k.to_dense().sum(dim=1)
            D_k_inv = torch.diag(D_k.pow(-1))
            D_k_inv = torch.where(torch.isinf(D_k_inv), torch.tensor(0.0), D_k_inv)
            hop_feature = torch.mm(A_k.to_dense(), D_k_inv @ data.x.float())
            hop_features.append(hop_feature)

        data.hop1_feature = hop_features[0]
        if len(self.K)==2:
            data.hop2_feature = hop_features[1]
        if "edge_attr" in data and self.agg=='sum':
                edge_feature = scatter_add(data.edge_attr, data.edge_index[1], dim=0, dim_size=data.num_nodes)
                data.edge_features = edge_feature
        return data


class KHopTransformScaled(BaseTransform):
    def __init__(self, K=[1, 2], agg='sum'):
        super(KHopTransformScaled, self).__init__()
        self.K = K
        self.agg = agg

    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        edge_index = data.edge_index
        
        hop_features = []
        for k in self.K:
            # Create adjacency matrix A and raise it to the power of k
            A_k = sp.coo_matrix((np.ones(edge_index.size(1)), edge_index.cpu().numpy()), shape=(N, N))
            A_k = A_k @ A_k if k == 2 else A_k  # if k > 2, need to multiply more times

            # Compute the row sum for A_k
            D_k = np.array(A_k.sum(axis=1)).squeeze()

            # Replace the dense inverse diagonal matrix D_k_inv with a sparse one
            D_k_inv = 1.0 / D_k
            D_k_inv[np.isinf(D_k_inv)] = 0

            # Create a sparse diagonal matrix with D_k_inv as the diagonal
            D_k_inv = diags(D_k_inv)

            # Multiply A_k with D_k_inv
            A_k = A_k @ D_k_inv

            # Convert back to PyTorch tensor
            A_k = torch.tensor(A_k.toarray(), device=device).float()

            hop_feature = A_k @ data.x.float()

            hop_features.append(hop_feature)

        data.hop1_feature = hop_features[0]
        if len(self.K) == 2:
            data.hop2_feature = hop_features[1]
        if "edge_attr" in data and self.agg=='sum':
            edge_feature = scatter_add(data.edge_attr, data.edge_index[1], dim=0, dim_size=data.num_nodes)
            data.edge_features = edge_feature
        return data


class KHopTransform_OGBMOL(BaseTransform):
    def __init__(self, K=[1], agg='sum'):
        super(KHopTransform_OGBMOL, self).__init__()
        self.K = K
        self.agg = agg

    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        edge_index = data.edge_index
        
        for k in self.K:
            # Create adjacency matrix A and raise it to the power of k
            A_k = sp.coo_matrix((np.ones(edge_index.size(1)), edge_index.cpu().numpy()), shape=(N, N))
            A_k = A_k @ A_k if k == 2 else A_k  # if k > 2, need to multiply more times

            # Compute the row sum for A_k
            D_k = np.array(A_k.sum(axis=1)).squeeze()

            # Replace the dense inverse diagonal matrix D_k_inv with a sparse one
            D_k_inv = 1.0 / D_k
            D_k_inv[np.isinf(D_k_inv)] = 0

            # Create a sparse diagonal matrix with D_k_inv as the diagonal
            D_k_inv = diags(D_k_inv)

            # Multiply A_k with D_k_inv
            A_k = A_k @ D_k_inv

            # Convert back to PyTorch tensor
            A_k = torch.tensor(A_k.toarray(), device=device).float()
            data.adjK = [[A_k]]
        if "edge_attr" in data and self.agg=='sum':
            edge_feature = scatter_add(data.edge_attr, data.edge_index[1], dim=0, dim_size=data.num_nodes)
            data.edge_features = edge_feature
        return data


class EdgeFeatureTransform(BaseTransform):
    def __init__(self):
        super(KHopTransform, self).__init__()

    def __call__(self, data):
        edge_neighbors = [[] for _ in range(data.num_nodes)]
        for edge, attr in zip(data.edge_index.T, data.edge_attr):
            edge_neighbors[edge[1].item()].append(attr)
        
        edge_neighbors = [torch.stack(n) for n in edge_neighbors if n]  # Convert to tensors
        data.edge_features = edge_neighbors
        return data

class SampleAndEncode(BaseTransform):
    def __init__(self, K, P): # P elements to sample with replacement
        self.K = K  # number of hops
        self.P = P # number of sampled nodes

    def __call__(self, data):
        N = data.num_nodes
        Khop_neighbors = data.Khop_neighbors
        X = data.x
        rw_feats = data.rw_matrix
        rw_dim = rw_feats.shape[1]
        num_feats = X.shape[1]
        final_output = []
        for j in range(self.K):
            node_output = []
            for v in range(N):
                val = Khop_neighbors[v]
                if len(val[j])>0:
                    sampled_elements = random.choices(val[j], k=self.P)
                    x = X[sampled_elements]
                    rw = rw_feats[sampled_elements]
                else:
                    x = torch.zeros(self.P,num_feats)
                    rw = torch.zeros(self.P,rw_dim)
                    
                x= x.unsqueeze(0).unsqueeze(0)
                rw = rw.unsqueeze(0).unsqueeze(0)
                one_hot_dist = F.one_hot(torch.tensor(j),self.K)
                one_hot_dist = one_hot_dist.repeat(self.P, 1)
                one_hot_dist = one_hot_dist.unsqueeze(0).unsqueeze(0)
                v = torch.cat((x,rw,one_hot_dist),dim=-1)
                node_output.append(v)
            v = torch.cat(node_output,dim=0)   # (N,1,P,D+RW+K)
            final_output.append(v)
        val = torch.cat(final_output,dim=1) # (N,K,P,D+RW+K)
        data.sampled_Khop_tensor=val
        return data


class DropEdge(BaseTransform):
    def __init__(self, p=0.08):
        """
        Initialize the transform. 
        p: The probability of an edge being dropped.
        """
        self.p = p

    def __call__(self, data):
        """
        Apply the transform. 
        data: A Data object, which includes edge_index attribute.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1) // 2  # Each edge appears twice

        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)

        # Create a mask of edges to drop
        mask = torch.rand(num_edges) < self.p
        drop_edges = edge_index[:, mask.repeat(2)].t().tolist()

        # Remove both directions of each dropped edge
        for i in range(0, len(drop_edges), 2):
            edge = drop_edges[i]
            reverse_edge = edge[::-1]
            edge_index = edge_index[:, ~((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))]
            edge_index = edge_index[:, ~((edge_index[0] == reverse_edge[0]) & (edge_index[1] == reverse_edge[1]))]

        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class CustomLaplacianEigenvectorPE:
    def __init__(self, attr_name=None,start=20):
        self.attr_name = attr_name
        self.start = start
    def __call__(self, data):
        num_feats = data.num_features
        for k in [self.start,10,5]:
            num_nodes = data.num_nodes
            if k>=num_nodes-1: 
                continue
            try:
                transform = AddLaplacianEigenvectorPE(k=k, attr_name=self.attr_name)
                data = transform(data)
                # Add zeros if fewer than 20 dimensions
                num_nodes = data.num_nodes
                remaining_dim = self.start - k
                if remaining_dim > 0:
                    zero_padding = torch.zeros((num_nodes, remaining_dim))
                    data.x = torch.cat((data.x, zero_padding), dim=-1)
                return data
            except:
                continue
                # print(f"Failed with k={k}, trying a smaller k. Error: {e}")


        # If all attempts failed, assign 20-dimensional all-zero vector
        # print("All attempts to calculate Laplacian Eigenvector failed. Assigning 20-d all-zero vector.")
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, self.start+num_feats))
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_current_datetime():
    # Get the current datetime
    now = datetime.now()

    # Format the datetime string
    datetime_str = now.strftime("%m%d")
    return datetime_str



def k_fold(dataset, folds):
    #! for TU Datasets
    skf = StratifiedKFold(folds, shuffle=True, random_state=1)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

if __name__ == '__main__':
    # Compose multiple transform in pytorch geometric
    dataset_name = "CIFAR10"
    print ("loading dataset")
    if dataset_name== "CIFAR10":
        pre_transforms = Compose([
            SuperpixelTransform(),
            ComputeKhopNeighbors(5),
            ComputeClusteringCoefficient(),
            PerformMetisClustering(n_clusters=5),
            PerformMetisClustering(n_clusters=10),
            PerformLouvainClustering()])
    else:
         pre_transforms = Compose([
            ComputeKhopNeighbors(5),
            PerformMetisClustering(n_clusters=5),
            PerformMetisClustering(n_clusters=10),
            PerformLouvainClustering()])       
    
    if dataset_name == "CIFAR10" or dataset_name == "PATTERN":
        dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',split='test',name=dataset_name,transform=PerformLouvainClustering())
    data = dataset[0]
    print (data)
    # # build a dataloader for data in pytorch geometric
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    # for t in dataloader:
    #     val = t
    #     print (val)    
    #     print (t.metis_clusters)
    #     break
    