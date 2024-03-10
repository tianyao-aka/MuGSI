from torch_geometric.datasets import GNNBenchmarkDataset
from model.data_utils import PerformLouvainClustering,PerformMetisClustering,SuperpixelTransform,CustomLaplacianEigenvectorPE,KHopTransform,AddAllOnes,KHopTransformScaled,KHopTransform_OGBMOL
from torch_geometric.transforms import BaseTransform,Compose,AddLaplacianEigenvectorPE
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from termcolor import colored
import argparse

def filter_num_nodes(data):

    return data.num_nodes >= 2


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default="ogbg-molhiv")
    args = parser.parse_args()
    args = vars(args)
    dataset_name = args['dataset']
    ## CIFAR10, PATTERN
    if 'CIFAR10' in args['dataset']:
        print (colored(f"loading dataset {dataset_name}",'blue','on_white'))
    
        print ("loading dataset") 

        dataset = GNNBenchmarkDataset(root=f'data_raw/{dataset_name}',name=dataset_name)    

        dataset = GNNBenchmarkDataset(root=f'data/GA_MLP/',name=dataset_name,pre_transform=Compose([SuperpixelTransform(),
                                                                                                    CustomLaplacianEigenvectorPE(),
                                                                                                    KHopTransform(),
                                                                                                    PerformLouvainClustering()
                                                                                                                ]))

        dl = DataLoader(dataset,batch_size=3,shuffle=False)
        print (dataset)
        print (dataset[0])

    
    
    ## OGBG
    if 'molhiv' in args['dataset']:
        print (colored(f"loading dataset {dataset_name}",'blue','on_white'))
        dataset = PygGraphPropPredDataset(root=f'data/raw/',name=dataset_name)
        dataset = PygGraphPropPredDataset(root=f'data/GA_MLP/',name=dataset_name,pre_transform=Compose([KHopTransform_OGBMOL(),
                                                                                                     PerformLouvainClustering()
                                                                                                             ]))
    


