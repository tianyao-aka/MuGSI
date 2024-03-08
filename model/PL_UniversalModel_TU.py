from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import nn
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
from termcolor import colored
from tqdm import tqdm
import os
from training_utils import *
from data_utils import write_results_to_file,save_model_weights,delete_folder
import warnings
from data_utils import get_current_datetime

warnings.filterwarnings("ignore")



class PL_UniversalModel_TU(pl.LightningModule):
    def __init__(self,model,lr=5e-3,weight_decay=1e-7,lr_patience=30,gamma=0.5,model_saving_path=None,test_loader=None,use_KD=False,**kargs):
        # only contains training and validation related parameters
        super(PL_UniversalModel_TU, self).__init__()
        self.save_hyperparameters(ignore=["model","val_loader","test_loader","teacherModel","pyg_dataset"])
        self.model = model
        self.lr=lr
        self.lr_patience=lr_patience
        self.gamma=gamma
        self.weight_decay=weight_decay
        self.num_classes = kargs['num_classes']
        self.dataset = kargs['dataset']
        self.fold_index = kargs["dataset_index"]
        self.teacherModel = kargs.get('teacherModel',None)
        
        self.acc = Accuracy(top_k=1,task='multiclass',num_classes=kargs['num_classes'])
        self.test_dataloader = test_loader

        self.val_acc=[]
        self.record_acc = []
        self.per_epoch_loss = []
        self.loss_cache = []
        self.best_valid_acc = -1.
        self.test_acc=0.
        self.max_epochs = kargs['max_epochs']
        self.seed = kargs['seed']
        
        self.num_hops = kargs["num_hops"]
        
        self.dropout = kargs['dropout']
        self.first_layer_dropout = 0.
        self.hidden_dim = kargs['hidden_dim']
        self.random_id = kargs['random_id']
        self.model_saving_path = model_saving_path
        self.loss_saving_path = kargs["loss_saving_path"]
        self.result_saving_path = kargs["result_saving_path"]
        self.device_id = kargs["device_id"]
        
        self.train_start_time = None
        self.train_end_time = None
        
        self.inference_time = []
        self.infer_start = 0.
        self.infer_end = 0.
        # for KD arguments in kargs
        
        self.use_KD = use_KD
        if use_KD:
            if kargs["studentModelName"] == "MLP":
                self.linear_proj = nn.Linear(2*kargs["hidden_dim"], kargs['teacher_hidden_dim'])
                if kargs["useNCE"]:
                    print ("useNCE",kargs["useNCE"])
                    self.nce_linear = nn.Linear(2*kargs["hidden_dim"], kargs['teacher_hidden_dim'])
            else:
                self.linear_proj = nn.Linear(2*kargs["hidden_dim"], kargs['teacher_hidden_dim'])
                if kargs["useNCE"]:
                    self.nce_linear = nn.Linear(2*kargs["hidden_dim"], kargs['teacher_hidden_dim'])                
            self.kl_div = nn.KLDivLoss(reduction='batchmean')
            # if use kd skills
            self.useSoftLabel = kargs.get("useSoftLabel",False)
            self.useNodeSim = kargs.get('useNodeSim',False)
            self.useNodeFeatureAlign = kargs.get('useNodeFeatureAlign',False)
            self.useClusterMatching = kargs.get('useClusterMatching',False)
            self.useRandomWalkConsistency = kargs.get('useRandomWalkConsistency',False)
            self.useDropoutEdge = kargs.get('useDropoutEdge',False)
            self.useMixUp = kargs.get('useMixUp',False)
            self.useGraphPooling = kargs.get('useGraphPooling',False)
            self.useNCE = kargs.get('useNCE',False)
            
            # strength of the KD skills
            self.softLabelReg = kargs.get('softLabelReg',1e-1)
            self.nodeSimReg = kargs.get('nodeSimReg',1e-3)
            self.NodeFeatureReg = kargs.get('NodeFeatureReg',0.0)
            self.ClusterMatchingReg = kargs.get('ClusterMatchingReg',0.0)
            self.clusterAlgo = kargs.get('clusterAlgo',"louvain")
            self.RandomWalkConsistencyReg = kargs.get('RandomWalkConsistencyReg',0.0)
            self.MixUpReg = kargs.get('MixUpReg',0.0)
            self.graphPoolingReg = kargs.get('graphPoolingReg',0.0)
            self.NCEReg = kargs.get('NCEReg',0.0)
            self.pathLength = kargs.get('pathLength',0)
            self.usePE = kargs.get('usePE',False)
            
            
            self.stu = kargs["studentModelName"]
            self.teacher= kargs["teacherModelName"]
            self.dataset = kargs["dataset"]
            
            self.expId = f"expId_{self.random_id}"
            self.use_AdditionalAttr = kargs.get("use_AdditionalAttr",False)
        else:
            self.stu = kargs["studentModelName"]
            self.teacher= kargs.get("teacherModelName",'empty')
            self.dataset = kargs["dataset"]
            
            self.expId = f"expId_{self.random_id}"
            self.use_AdditionalAttr = kargs.get("use_AdditionalAttr",False)           



    def forward(self, data):
        x = self.model(data)
        return x

    def on_train_epoch_start(self) -> None:
        self.train_start_time = time()
    
    
    def training_step(self, batch, batch_idx):
        # x, y,batch = batch.x,batch.y, batch.batch
        y = batch.y
        y = y.view(-1,)
        if self.use_KD:
            y_pred,node_emb,graph_emb = self.model(batch,output_emb =True) # logits after graph readout function
        else:
            y_pred = self.model(batch)
        
        if self.num_classes > 1: # multiple classes, not binary classification
            # Classification problem: use log softmax loss
            loss = F.nll_loss(F.log_softmax(y_pred, dim=1), y)
            self.loss_cache.append(loss.detach().item())
        
        
        if not self.use_KD:
            self.log('train_loss', loss,prog_bar=True,on_epoch=True)
            return loss
        else:
            total_loss = loss
            nodeSimLoss= 0.0
            softLabelLoss = 0.0
            featureAlignmentLoss = 0.0
            graphPoolingLoss = 0.0
            mixUpLoss = 0.0
            clusterMatchingLoss = 0.0
            graphAlignmentLoss = 0.0
            
            
            teacherPred = batch.teacherPred
            if self.useSoftLabel:
                softLabelLoss = calc_KL_divergence(y_pred,teacherPred,self.kl_div)
                total_loss = total_loss +  self.softLabelReg*softLabelLoss
            
            if self.useNodeSim:
                teacherNodeEmb = batch.nodeEmb  # teacher node embeddings
                nodeSimLoss = calc_node_similarity(node_emb,teacherNodeEmb)
                total_loss = total_loss + self.nodeSimReg*nodeSimLoss
            if self.useNodeFeatureAlign:
                val = self.linear_proj(node_emb)
                teacherNodeEmb = batch.nodeEmb
                featureAlignmentLoss = nodeFeatureAlignment(val,teacherNodeEmb)
                total_loss = total_loss +  self.NodeFeatureReg*featureAlignmentLoss
            
            if self.useRandomWalkConsistency:
                rwLoss = 0.0
                num_graphs = batch.num_graphs
                batch_idx = batch.batch
                teacherNodeEmb = batch.nodeEmb
                random_walk_paths = batch.random_walk_paths
                # ! Need to consider per-graph cond probabilities. use for loop and batch.batch to get each graph
                for idx,i in enumerate(range(num_graphs)):
                    mask_graph = batch_idx == i
                    if torch.all(random_walk_paths[20*idx:20*(idx+1)]==1):
                        continue
                    teacherCondProb = calculate_conditional_probabilities(random_walk_paths[20*idx:20*(idx+1)],teacherNodeEmb[mask_graph])  #! sample_size for random_walk per graph is 20
                    stuCondProb = calculate_conditional_probabilities(random_walk_paths[20*idx:20*(idx+1)],node_emb[mask_graph])
                    rwWalkLoss_per_graph = calculate_kl_loss(stuCondProb,teacherCondProb)
                    rwLoss = rwLoss + rwWalkLoss_per_graph
                rwLoss = rwLoss/num_graphs
                total_loss = total_loss +  self.RandomWalkConsistencyReg*rwLoss
                
                # print (colored(f"loss:{loss}, rwWalkLoss: {rwWalkLoss}",'green','on_red'))
            if self.useGraphPooling:
                teacherGraphEmb = batch.graphEmb
                val = self.linear_proj(graph_emb)
                graphAlignmentLoss = nodeFeatureAlignment(val,teacherGraphEmb)  # nodeFeatureAlignment can also be used for graph embedding matching
                total_loss = total_loss +  self.graphPoolingReg*graphAlignmentLoss
                
            if self.useClusterMatching:
                num_graphs = batch.num_graphs
                batch_idx = batch.batch
                h=[]
                cluster_num = []
                for i in range(num_graphs):
                    if self.clusterAlgo=='louvain':
                        cluster_id = batch.louvain_cluster_id.view(-1,)
                    if self.clusterAlgo=='metis5':
                        cluster_id = batch.metis_clusters5.view(-1,)
                    if self.clusterAlgo=='metis10':
                        cluster_id = batch.metis_clusters10.view(-1,)
                    mask_graph = batch_idx == i
                    cluster_id_per_graph = cluster_id[mask_graph]
                    
                    node_emb_per_graph = node_emb[mask_graph]
                    h.append(self.model.pool(node_emb_per_graph,cluster_id_per_graph))
                    cluster_num.append(cluster_id_per_graph.max()+1)
                teacherClusterInfo = batch.teacherClusterInfo
                splitted_tensor = torch.split(teacherClusterInfo,cluster_num)
                for i,t in enumerate(splitted_tensor):
                    if t.shape[0]<=2: continue
                    cos_sim_teacher = fast_cosine_sim_matrix(t,t)
                    cos_sim_stu = fast_cosine_sim_matrix(h[i],h[i])
                    val = (cos_sim_stu-cos_sim_teacher).norm(p='fro')**2/2.0
                    if torch.isnan(val) or torch.isinf(val):
                        continue
                    clusterMatchingLoss += val
                    # print (f"clusterMatchingLoss:{clusterMatchingLoss}.....")
                total_loss += clusterMatchingLoss*self.ClusterMatchingReg/num_graphs
            
            loss_dict = {'train_loss':total_loss}
            self.log_dict(loss_dict,prog_bar=True,on_epoch=True)
            return total_loss
        

    def validation_step(self, batch, batch_idx):
        tot_correct = 0.
        _ , y, _ = batch.x, batch.y, batch.batch
        y_pred = self.model(batch)
        y_pred = torch.argmax(y_pred, dim=1)
        tot_correct += torch.sum(y.to('cpu') == y_pred.to('cpu')).item()
        N = len(y)
        self.val_acc.append((N,tot_correct))


    def on_validation_epoch_end(self):
        tot_correct = sum([i[1] for i in self.val_acc])
        N = sum([i[0] for i in self.val_acc])
        val_acc = 1.*tot_correct/N
        self.log('valid_acc', val_acc, prog_bar=True, on_epoch=True)
        self.val_acc = []
        if self.current_epoch<=1:
            self.best_valid_acc = 0.0
            return 
        if val_acc > self.best_valid_acc:
            self.best_valid_acc = val_acc
            if self.current_epoch==0:
                self.best_valid_acc = 0.0
            if self.model_saving_path is not None:
                file_name = f"BestValidAccuracy_{self.best_valid_acc:5f}_model_weights.pt"
                if self.model_saving_path is not None:
                    save_model_weights(self.model,self.model_saving_path,file_name=file_name)
            if self.use_KD:
                info = f"Dataset:{self.dataset},fold_index:{self.fold_index}, Seed:{self.seed}, expId:{self.expId},student:{self.stu}, teacher:{self.teacher}, useSoftLabel:{self.useSoftLabel}, useNodeSim:{self.useNodeSim},useLearnableGraphPooling:{self.useGraphPooling}, useNodeFeatureAlign:{self.useNodeFeatureAlign},useClusterMatching: {self.useClusterMatching}, useRandomWalkConsistency:{self.useRandomWalkConsistency}, useDropoutEdge:{self.useDropoutEdge}, useNCE:{self.useNCE}, "
                info2 = f"softLabelReg:{self.softLabelReg}, nodeSimReg:{self.nodeSimReg}, NodeFeatureReg:{self.NodeFeatureReg}, ClusterMatchingReg:{self.ClusterMatchingReg}, graphPoolingReg:{self.graphPoolingReg}, pathLength:{self.pathLength}, RandomWalkConsistencyReg:{self.RandomWalkConsistencyReg}, NCEReg:{self.NCEReg},use_AdditionalAttr:{self.use_AdditionalAttr}, BestValidAccuracy: {self.best_valid_acc:6f}"
                info = info+info2
                print (colored(f'writing to path:{self.result_saving_path}','blue','on_white'))
                write_results_to_file(f"{self.result_saving_path}",f"{self.dataset}_student_{self.stu}_teacher_{self.teacher}_expId_{self.random_id}.txt",info)
            else:
                info = f"Not using KD, Seed:{self.seed}, ExpId:{self.random_id},model name:{self.stu},num hops:{self.num_hops}, BestTestAccuracy: {self.best_valid_acc:6f}"
                write_results_to_file(f"{self.result_saving_path}",f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt",info)                
            
                
        print (colored(f'Seed:{self.seed}, dataset:{self.dataset},using device:{self.device_id} BestValidationAccuracy at epoch:{self.current_epoch} is {self.best_valid_acc:6f}', 'red','on_yellow'))
        if not self.use_KD:
            if self.current_epoch== self.max_epochs-1:
                info = f"Not using KD, Seed:{self.seed}, ExpId:{self.random_id},model name:{self.stu},num hops:{self.num_hops}, BestTestAccuracy: {self.best_valid_acc:6f}"
                write_results_to_file(f"{self.result_saving_path}",f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt",info)
            print (colored(f'Dataset:{self.dataset},fold:{self.fold_index}, Not using KD,Seed:{self.seed}, ExpId:{self.random_id},model name:{self.stu},num hops:{self.num_hops}, TestAccuracy at epoch:{self.current_epoch} is {self.best_valid_acc:6f}:','red','on_yellow'))
        else:
            info = f"Dataset:{self.dataset},fold_index:{self.fold_index}, Seed:{self.seed}, expId:{self.expId},student:{self.stu}, teacher:{self.teacher}, useSoftLabel:{self.useSoftLabel}, useNodeSim:{self.useNodeSim},useLearnableGraphPooling:{self.useGraphPooling}, useNodeFeatureAlign:{self.useNodeFeatureAlign},useClusterMatching: {self.useClusterMatching}, useRandomWalkConsistency:{self.useRandomWalkConsistency}, useDropoutEdge:{self.useDropoutEdge}, useNCE:{self.useNCE}, "
            info2 = f"softLabelReg:{self.softLabelReg}, nodeSimReg:{self.nodeSimReg}, NodeFeatureReg:{self.NodeFeatureReg}, ClusterMatchingReg:{self.ClusterMatchingReg}, graphPoolingReg:{self.graphPoolingReg}, pathLength:{self.pathLength}, RandomWalkConsistencyReg:{self.RandomWalkConsistencyReg}, NCEReg:{self.NCEReg},use_AdditionalAttr:{self.use_AdditionalAttr}, BestValidAccuracy: {self.best_valid_acc:6f}"
            info = info+info2
            print (colored(info,'blue','on_white'))

            if self.current_epoch== self.max_epochs-1:
                write_results_to_file(f"{self.result_saving_path}",f"{self.dataset}_student_{self.stu}_teacher_{self.teacher}_expId_{self.random_id}.txt",info)

 
 
    def test(self):
        print (colored("----------------Testing Stage---------------","blue","on_white"))
        self.eval()
        tot_correct = 0.
        y_preds = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                batch = batch.to(self.device)
                _, y, _ = batch.x, batch.y, batch.batch
                y_pred = self.model(batch)
                labels.append(y.view(-1,))
                y_preds.append(y_pred)
        labels = torch.cat(labels, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_preds = torch.argmax(y_preds, dim=1)
        tot_correct += torch.sum(labels == y_preds).item()
        test_acc = 1.*tot_correct/len(labels)
        self.log('test_acc', test_acc, prog_bar=True, on_epoch=True)
        print (colored(f'expId:{self.random_id},TestAccuracy at epoch:{self.current_epoch} is: {test_acc}', 'red','on_yellow'))
        self.train()
        self.test_acc=test_acc
        return test_acc


    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=8e-3, weight_decay=self.weight_decay)
        # We will reduce the learning rate 
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.6,patience=self.lr_patience,min_lr=5e-7)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'valid_acc'}
        # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.lr_patience,gamma=self.gamma)
        # return {'optimizer':optimizer,'lr_scheduler':scheduler}


if __name__ =='__main__':
    a = torch.randn(50, 10)
    b = torch.randn(50, 10)




