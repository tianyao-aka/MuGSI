import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric import transforms as T
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from time import time
from torch_geometric.transforms import BaseTransform,Compose
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
from pynvml import *
import shutil
import glob
from difflib import SequenceMatcher

def filter_result_df(df, query_keys):
    similar_cols = []

    for query_key in query_keys:
        # Identify column most similar to query key
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        reg_cols = [col for col in df.columns if 'Reg' in col]
        similar_col = max(reg_cols, key=lambda col: similar(query_key, col))
        similar_cols.append(similar_col)

    # Keep rows where only the query keys are 'True' and non-query keys are 'False'
    use_cols = [col for col in df.columns if col.startswith('use')]
    non_query_use_cols = [col for col in use_cols if col not in query_keys and "Additional" not in col]
    
    # Use boolean arrays to filter rows
    condition_query_keys = df[query_keys] == "True"
    condition_non_query_use_cols = df[non_query_use_cols] == "False"
    df = df[(condition_query_keys.all(axis=1)) & (condition_non_query_use_cols.all(axis=1))]

    # Remove other columns containing 'use' and 'Reg' in their name
    df = df.drop(columns=[col for col in df.columns if col.startswith('use') and col not in query_keys])
    df = df.drop(columns=[col for col in reg_cols if col not in similar_cols])

    return df





def results_to_df(strings):
    data = []
    for string in strings:
        dict_data = {}
        pairs = string.split(",")
        for pair in pairs:
            key, value = pair.split(":")
            key = key.strip()
            value = value.strip()

            # Try to convert numeric values into float
            try:
                value = float(value)
            except ValueError:
                # If conversion to float fails, it must be a string value
                pass

            dict_data[key] = value
        data.append(dict_data)

    # Create the DataFrame
    df = pd.DataFrame.from_records(data)
    return df

def read_results(directory):
    files_content = []
    for filename in glob.iglob(directory + '**/*.txt', recursive=True):
        with open(filename, 'r') as file:
            files_content.append(file.read())
    return files_content

def list_to_df(lst):
    # Initialize empty dictionaries for each column
    data = {'seed': [], 'useNodeSim': [], 'softLabelReg': [], 'nodeSimReg': [], 'TestAccuracy': []}

    # Loop through each entry in the list
    for item in lst:
        # Initialize dictionary for each row
        row = {'seed': None, 'useNodeSim': None, 'softLabelReg': None, 'nodeSimReg': None, 'TestAccuracy': None}

        # Split the string by commas
        items = item.split(", ")

        # Loop through each split item
        for i in items:
            # Split the item by colon
            sub_items = i.split(":")
            field_name = sub_items[0]

            # Check which column the item belongs to and append it to the correct list
            if field_name == 'Seed':
                row['seed'] = int(sub_items[1])
            elif field_name == 'useNodeSim':
                row['useNodeSim'] = sub_items[1] == 'True'
            elif field_name == 'softLabelReg':
                row['softLabelReg'] = float(sub_items[1])
            elif field_name == 'nodeSimReg':
                row['nodeSimReg'] = float(sub_items[1])
            elif 'TestAccuracy' in field_name:
                # Strip leading/trailing whitespace, split by space and get accuracy value
                TestAccuracy = sub_items[1].strip().split(' ')[-1]
                row['TestAccuracy'] = float(TestAccuracy)
                
        # Add row to data
        for key in row:
            data[key].append(row[key])
    # Combine all lists into a dataframe
    df = pd.DataFrame(data)
    return  df.dropna()



