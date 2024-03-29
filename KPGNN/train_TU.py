"""
script to train on TU dataset with GIN setting:https://github.com/weihua916/powerful-gnns
"""
import argparse
import os
import shutil
import time
from itertools import product
from json import dumps

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.seed import seed_everything
from tqdm import tqdm

import KPGNN.train_utils as train_utils
from KPGNN.data_utils import extract_multi_hop_neighbors, resistance_distance, post_transform
from KPGNN.datasets.tu_dataset import TUDatasetGINSplit, TUDataset
from KPGNN.layers.input_encoder import LinearEncoder
from KPGNN.layers.layer_utils import make_gnn_layer
from KPGNN.models.GraphClassification import GraphClassification
from KPGNN.models.model_utils import make_GNN


# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def train_TU(loader, model, optimizer, device, parallel=False):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        out = model(data)
        out = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_TU(loader, model, device, parallel=False):
    model.eval()
    loss = 0
    correct = 0
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        out = model(data)
        pred = out.max(1)[1]
        out = F.log_softmax(out, dim=-1)
        loss += F.nll_loss(out, y.view(-1), reduction='sum').item()
        correct += pred.eq(y.view(-1)).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def cross_validation_GIN_split(dataset, args, device, loader, log=None):
    """Cross validation framework with GIN split.
    Args:
        dataset(PyG.dataset): PyG dataset for training and testing
        args(Namesapce): arguments parser
        device(str): training device
        loader (DataLoader): dataloader for model training
        log(logger): log file
    """
    
    folds = 10
    lr_decay_step_size = 50
    test_losses, accs, durations = [], [], []
    count = 1
    k_fold_indices = dataset.train_indices, dataset.test_indices
    for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
        print("CV fold " + str(count))
        count += 1
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = loader(train_dataset, args.batch_size, shuffle=True)
        test_loader = loader(test_dataset, args.batch_size, shuffle=False)

        model = get_model(args)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
        t_start = time.perf_counter()

        pbar = tqdm(range(1, args.num_epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train_TU(train_loader, model, optimizer, device, parallel=args.parallel)
            test_loss, test_acc = val_TU(test_loader, model, device, parallel=args.parallel)
            test_losses.append(test_loss)
            accs.append(test_acc)
            eval_info = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            info = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold + 1, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            log.info(info)

            # decay the learning rate
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.factor * param_group["lr"]

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = torch.tensor(test_losses), torch.tensor(accs), torch.tensor(durations)
    loss, acc = loss.view(folds, args.num_epochs), acc.view(folds, args.num_epochs)
    acc_max, _ = acc.max(1)
    acc_mean = acc.mean(0)
    acc_cross_epoch_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    info = ('Test Loss: {:.4f}, Test Max Accuracy:{:.3f} ± {:.3f}, Test Max Cross Epoch Accuracy: {:.3f} ± {:.3f}, ' +
            'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.mean().item(),
        acc_max.std().item(),
        acc_cross_epoch_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    log.info(info)

    return (acc_max.mean().item(), acc_max.std().item()), \
           (acc_cross_epoch_max.item(), acc[:, argmax].std().item()), \
           (acc_final.item(), acc[:, -1].std().item())


def cross_validation_with_PyG_dataset(dataset, args, device, loader, log=None, seed=234):
    """Cross validation framework without validation dataset. Adapted from Nested GNN:https://github.com/muhanzhang/NestedGNN
    Args:
        dataset(PyG.dataset): PyG dataset for training and testing
        args(Namesapce): arguments parser
        device(str): training device
        loader (DataLOader): dataloader for model training
        log(logger): log file
        seed(int): random seed
    """
    folds = 10
    lr_decay_step_size = 50
    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*train_utils.k_fold(dataset, folds, seed))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = loader(train_dataset, args.batch_size, shuffle=True)
        test_loader = loader(test_dataset, args.batch_size, shuffle=False)

        model = get_model(args)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
        t_start = time.perf_counter()

        pbar = tqdm(range(1, args.num_epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train_TU(train_loader, model, optimizer, device, parallel=args.parallel)
            test_loss, test_acc = val_TU(test_loader, model, device, parallel=args.parallel)
            test_losses.append(test_loss)
            accs.append(test_acc)
            eval_info = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            info = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold + 1, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            log.info(info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.factor * param_group["lr"]

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = torch.tensor(test_losses), torch.tensor(accs), torch.tensor(durations)
    loss, acc = loss.view(folds, args.num_epochs), acc.view(folds, args.num_epochs)
    acc_max, _ = acc.max(1)
    acc_mean = acc.mean(0)
    acc_cross_epoch_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    info = ('Test Loss: {:.4f}, Test Max Accuracy:{:.3f} ± {:.3f}, Test Max Cross Epoch Accuracy: {:.3f} ± {:.3f}, ' +
            'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.mean().item(),
        acc_max.std().item(),
        acc_cross_epoch_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    log.info(info)

    return (acc_max.mean().item(), acc_max.std().item()), \
           (acc_cross_epoch_max.item(), acc[:, argmax].std().item()), \
           (acc_final.item(), acc[:, -1].std().item())


def get_model(args):
    layer = make_gnn_layer(args)
    init_emb = LinearEncoder(args.input_size, args.hidden_size)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob)

    model = GraphClassification(embedding_model=gnn,
                                pooling_method=args.pooling_method,
                                output_size=args.output_size)

    model.reset_parameters()

    if args.parallel:
        model = DataParallel(model, args.gpu_ids)
    return model


def edge_feature_transform(data):
    if data.edge_attr is not None:
        data.edge_attr = torch.where(data.edge_attr == 1)[1] + 2
    return data


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="MUTAG",
                        choices=("MUTAG", "DD", "PROTEINS", "PTC", "IMDBBINARY"), help='Name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.5,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument("--parallel", action="store_true",
                        help="If true, use DataParallel for multi-gpu training")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--l2_wd', type=float, default=3e-4, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="gd", choices=("gd", "spd"),
                        help="The kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=350, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGIN",
                        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Base GNN model")
    parser.add_argument("--K", type=int, default=2, help="Number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=30,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=1,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=30,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=5,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=50,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='If true, remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='If true, remove peripheral node configuration information from model')
    parser.add_argument('--wo_path_encoding', action='store_true',
                        help='If true, remove path encoding information from model')
    parser.add_argument('--wo_edge_feature', action='store_true',
                        help='If true, remove edge feature from model')
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last", "concat"),
                        help="Jumping knowledge method")
    parser.add_argument("--residual", action="store_true", help="If true, use residual connection between each layer")
    parser.add_argument("--use_rd", action="store_true", help="If true, add resistance distance feature to model")
    parser.add_argument("--virtual_node", action="store_true",
                        help="If true, add virtual node information in each layer")
    parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN")
    parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon in GIN model is trainable")
    parser.add_argument("--combine", type=str, default="geometric", choices=("attention", "geometric"),
                        help="Combine method in k-hop aggregation")
    parser.add_argument("--pooling_method", type=str, default="sum", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph classification")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                        help="Normalization method in model")
    parser.add_argument('--aggr', type=str, default="add",
                        help='Aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--factor', type=float, default=0.5, help='Factor for reducing learning rate scheduler')
    parser.add_argument('--reprocess', action="store_true", help='If true, reprocess the dataset')
    parser.add_argument('--search', action="store_true", help='If true, search hyper-parameters')

    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + \
                str(args.wo_edge_feature) + "_" + str(args.search)
    # Set up logging and devices
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, type=args.dataset_name)
    log = train_utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = train_utils.get_available_devices()
    if len(args.gpu_ids) > 1 and args.parallel:
        log.info(f'Using multi-gpu training')
        args.parallel = True
        loader = DataListLoader
        args.batch_size *= max(1, len(args.gpu_ids))
    else:
        log.info(f'Using single-gpu training')
        args.parallel = False
        loader = DataLoader

    # Set random seed
    seed = train_utils.get_seed(args.seed)
    log.info(f'Using random seed {seed}...')
    seed_everything(seed)

    def multihop_transform(g):
        return extract_multi_hop_neighbors(g, args.K, args.max_pe_num, args.max_hop_num, args.max_edge_type,
                                           args.max_edge_count,
                                           args.max_distance_count, args.kernel)

    if args.use_rd:
        rd_feature = resistance_distance
    else:
        def rd_feature(g):
            return g

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    if args.search:
        log.info("----------------------------------------------------------------")
        kernels = ["spd", "gd"]
        Ks = [2, 3, 4]
        layers = [2, 3, 4]
        combines = ["geometric", "attention"]
        t = product(kernels, Ks, layers, combines)
        best_graphSNN_result = (0, 1e30)
        best_gin_result = (0, 1e30)
        best_final_result = (0, 1e30)
        try:
            for parameters in t:
                kernel, K, layer, combine = parameters
                args.combine = combine
                args.kernel = kernel
                args.K = K
                args.num_layer = layer
                if K == 3:
                    args.hidden_size = 33
                else:
                    args.hidden_size = 32

                log.info(f"Search on kernel:{kernel},K:{K},layer:{layer},combine:{combine}")
                path = "data/TUGIN_"
                path = path + str(args.K) + "_" + args.kernel
                if os.path.exists(path + "/" + args.dataset_name + '/processed') and args.reprocess:
                    shutil.rmtree(path + "/" + args.dataset_name + '/processed')
                if args.dataset_name == "DD":
                    dataset = TUDataset(path, args.dataset_name,
                                        pre_transform=T.Compose(
                                            [edge_feature_transform, multihop_transform, rd_feature]),
                                        transform=transform,
                                        cleaned=False)
                else:
                    dataset = TUDatasetGINSplit(args.dataset_name, path,
                                                pre_transform=T.Compose(
                                                    [edge_feature_transform, multihop_transform, rd_feature]),
                                                transform=transform)

                args.input_size = dataset.num_node_features
                args.output_size = dataset.num_classes

                if args.dataset_name == "DD":
                    graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_with_PyG_dataset(
                        dataset, args, device, loader, log=log)
                else:
                    graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_GIN_split(
                        dataset, args, device, loader, log=log)

                if graphsnn_setting_result[0] > best_graphSNN_result[0] or \
                        (graphsnn_setting_result[0] == best_graphSNN_result[0] and graphsnn_setting_result[0] <
                         best_graphSNN_result[0]):
                    best_graphSNN_result = graphsnn_setting_result
                    best_graphSNN_paras = (kernel, K, layer, combine)

                if gin_setting_result[0] > best_gin_result[0] or \
                        (gin_setting_result[0] == best_gin_result[0] and gin_setting_result[0] < best_gin_result[0]):
                    best_gin_result = gin_setting_result
                    best_gin_paras = (kernel, K, layer, combine)

                if final_epoch_result[0] > best_final_result[0] or \
                        (final_epoch_result[0] == best_final_result[0] and final_epoch_result[0] < best_final_result[
                            0]):
                    best_final_result = final_epoch_result
                    best_final_paras = (kernel, K, layer, combine)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')

        log.info("----------------------------Complete search-----------------------------------")
        desc = '{:.3f} ± {:.3f}'.format(
            best_graphSNN_result[0], best_graphSNN_result[1]
        )
        log.info(f'Best result on graphSNN setting - {desc}, '
                 f'with {str(best_graphSNN_paras[0])} kernel,'
                 f'{str(best_graphSNN_paras[1])} hop, '
                 f'{str(best_graphSNN_paras[2])} layers, '
                 f'and {str(best_graphSNN_paras[3])} combination')
        log.info("===============================================================================")
        desc = '{:.3f} ± {:.3f}'.format(
            best_gin_result[0], best_gin_result[1]
        )
        log.info(f'Best result on GIN setting - {desc}, '
                 f'with {str(best_gin_paras[0])} kernel,'
                 f'{str(best_gin_paras[1])} hop, '
                 f'{str(best_gin_paras[2])} layers, '
                 f'and {str(best_gin_paras[3])} combination')
        log.info("===============================================================================")
        desc = '{:.3f} ± {:.3f}'.format(
            best_final_result[0], best_final_result[1]
        )
        log.info(f'Best result on final epoch - {desc}, '
                 f'with {str(best_final_paras[0])} kernel,'
                 f'{str(best_final_paras[1])} hop, '
                 f'{str(best_final_paras[2])} layers, '
                 f'and {str(best_final_paras[3])} combination')

    else:
        path = "data/TUGIN_"
        path = path + str(args.K) + "_" + args.kernel

        if os.path.exists(path + "/" + args.dataset_name + '/processed') and args.reprocess:
            shutil.rmtree(path + "/" + args.dataset_name + '/processed')
        if args.dataset_name == "DD":
            dataset = TUDataset(path, args.dataset_name,
                                pre_transform=T.Compose([edge_feature_transform, multihop_transform, rd_feature]),
                                transform=transform,
                                cleaned=False)

        else:
            dataset = TUDatasetGINSplit(args.dataset_name, path,
                                        pre_transform=T.Compose(
                                            [edge_feature_transform, multihop_transform, rd_feature]),
                                        transform=transform)

        args.input_size = dataset.num_node_features
        args.output_size = dataset.num_classes

        if args.dataset_name == "DD":
            graphsnn_setting_result, gin_setting_result, final_epoch_result = \
                cross_validation_with_PyG_dataset(dataset, args, device, loader, log=log)
        else:
            graphsnn_setting_result, gin_setting_result, final_epoch_result = \
                cross_validation_GIN_split(dataset, args, device, loader, log=log)


if __name__ == "__main__":
    main()
