import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# write a class
def DatasetVars(name = "CIFAR10",teacherName="GIN"):
    if teacherName=="GIN":    
        if name == "CIFAR10":
            return {"num_classes": 10,"num_features": 5,"teacher_hidden_dim":32}
        elif "IMDB-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,"teacherPath":"best_models/IMDB-BINARY/GIN/hidden_64_dropout_0.0_num_layers_5/","hidden":64,"dropout":0.0,"num_layers":5}
        elif "REDDIT-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,"teacherPath":"best_models/REDDIT-BINARY/GIN/hidden_16_dropout_0.0_num_layers_5/","hidden":16,"dropout":0.0,"num_layers":5}    
        elif name == "PROTEINS":
            return {"num_classes": 2,"num_features": 3,"teacherPath":"best_models/PROTEINS/GIN/hidden_64_dropout_0.0_num_layers_5/","hidden":64,"dropout":0.0,"num_layers":5}
        elif name == "ENZYMES":
            return {"num_classes": 6,"num_features": 3,"teacherPath":"best_models/ENZYMES/GIN/hidden_64_dropout_0.0_num_layers_3/","hidden":64,"dropout":0.0,"num_layers":3}
        elif name == "NCI1":
            return {"num_classes": 2,"num_features": 37,"teacherPath":"best_models/NCI1/GIN/hidden_64_dropout_0.0_num_layers_5/","hidden":64,"dropout":0.0,"num_layers":5}
        elif name=="BZR":
            return {"num_classes": 2,"num_features": 53,"teacherPath":"best_models/BZR/GIN/hidden_64_dropout_0.0_num_layers_3_hops_2_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":3}
        elif name=="DD":
            return {"num_classes": 2,"num_features": 89,"teacherPath":"best_models/DD/GIN/hidden_64_dropout_0.0_num_layers_5_hops_2_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":5}
        if "MUTAG" in name:
            return {"num_classes": 2,"num_features": 7,"teacherPath":"best_models/BZR/GCN/hidden_64_dropout_0.0_num_layers_3_hops_1_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":3}
    if teacherName=="GCN":
        if "REDDIT-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,"teacherPath":"best_models/REDDIT-BINARY/GCN/hidden_64_dropout_0.0_num_layers_5_hops_1_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":5}   
        if "PROTEINS" in name:
            return {"num_classes": 2,"num_features": 3,"teacherPath":"best_models/PROTEINS/GCN/hidden_64_dropout_0.0_num_layers_5_hops_1_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":5}
        if "DD" in name:
            return {"num_classes": 2,"num_features": 89,"teacherPath":"best_models/DD/GCN/hidden_64_dropout_0.5_num_layers_5_hops_1_batch_size_32/","hidden":64,"dropout":0.5,"num_layers":5}
        if "BZR" in name:
            return {"num_classes": 2,"num_features": 53,"teacherPath":"best_models/BZR/GCN/hidden_64_dropout_0.0_num_layers_3_hops_1_batch_size_32/","hidden":64,"dropout":0.0,"num_layers":3}
        if "IMDB-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,"teacherPath":"best_models/IMDB-BINARY/GCN/hidden_64_dropout_0.5_num_layers_3_hops_1_batch_size_32/","hidden":64,"dropout":0.5,"num_layers":3}
    if teacherName =='KPGNN' or "KP" in teacherName: 
        if "PROTEINS" in name: 
            return {"num_classes": 2,"num_features": 3,"teacherPath":"best_models/PROTEINS/KPGIN/hidden_66_dropout_0.5_num_layers_4_hops_3_batch_size_32_kernel_spd_combine_attention/","hidden":66,"dropout":0.5,"num_layers":4,'K':3,'kernel':'spd','combine':'attention'}
        if "DD" in name: 
            return {"num_classes": 2,"num_features": 89,"teacherPath":"best_models/DD/KPGIN/hidden_66_dropout_0.5_num_layers_3_hops_3_batch_size_32_kernel_spd_combine_attention/","hidden":66,"dropout":0.5,"num_layers":3,'K':3,'kernel':'spd','combine':'attention'}
        if "BZR" in name:
            return {"num_classes": 2,"num_features": 53,"teacherPath":"best_models/BZR/KPGIN/hidden_64_dropout_0.0_num_layers_4_hops_4_batch_size_32_kernel_spd_combine_attention/","hidden":64,"dropout":0.0,"num_layers":4,'K':4,'kernel':'spd','combine':'attention'}        
        if "IMDB-BINARY" in name: 
            return {"num_classes": 2,"num_features": 65,"teacherPath":"best_models/IMDB-BINARY/KPGIN/hidden_64_dropout_0.5_num_layers_3_hops_4_batch_size_32_kernel_spd_combine_geometric/","hidden":64,"dropout":0.5,"num_layers":3,'K':4,'kernel':'spd','combine':'geometric'}


