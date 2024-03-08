from torch_geometric.datasets import GNNBenchmarkDataset,TUDataset
from model.data_utils import PerformLouvainClustering,PerformMetisClustering,SuperpixelTransform,KHopTransform,CustomLaplacianEigenvectorPE,DegreeTransform
from torch_geometric.transforms import BaseTransform,Compose
from torch_geometric.loader import DataLoader
# from ogb.graphproppred import PygGraphPropPredDataset
from termcolor import colored
import argparse
import warnings
from glob import glob

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default="IMDB-BINARY")
    args = parser.parse_args()
    args = vars(args)
    dataset_name = args['dataset']
    # ! MLP
    print (colored(f"loading dataset {dataset_name}",'blue','on_white'))
    if "IMDB" in dataset_name or "REDDIT" in dataset_name or "COLLAB" in dataset_name:
        dataset = TUDataset(root=f'data/raw/',name=dataset_name,pre_transform=DegreeTransform())
        print ("raw:",dataset[0])
        dataset = TUDataset(root=f'data/withAdditionalAttr/',name=dataset_name,pre_transform=Compose([DegreeTransform(),
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                PerformLouvainClustering()]))
    else:
        dataset = TUDataset(root=f'data/withAdditionalAttr/',name=dataset_name,pre_transform=Compose([
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                PerformLouvainClustering()]))            
        dataset = TUDataset(root=f'data/raw/',name=dataset_name)
        print ("raw:",dataset[0])
    
    #! GA-MLP
    print (colored(f"loading dataset {dataset_name}",'blue','on_white'))
    if "IMDB" in dataset_name or "REDDIT" in dataset_name or "COLLAB" in dataset_name:
        dataset = TUDataset(root=f'data/raw/',name=dataset_name,pre_transform=DegreeTransform())
        print ("raw:",dataset[0])
        dataset = TUDataset(root=f'data/GA_MLP/',name=dataset_name,pre_transform=Compose([DegreeTransform(),
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                KHopTransform(),
                                                                                                PerformLouvainClustering()]))
    else:
        dataset = TUDataset(root=f'data/GA_MLP/',name=dataset_name,pre_transform=Compose([
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                KHopTransform(),
                                                                                                PerformLouvainClustering()]))            
        print ("ga-mlp:",dataset[0])
        dataset = TUDataset(root=f'data/raw/',name=dataset_name)
        
        
        