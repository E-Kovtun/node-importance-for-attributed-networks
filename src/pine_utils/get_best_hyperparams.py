from src.data_preparation.prepare_graph import get_graph
from src.data_preparation.split_data import transductive_edge_split
from src.backbone.gat_model import GAT_PINE
from src.pine_utils.train_lp import train
from src.pine_utils.eval_lp import test
from tqdm import tqdm
import json 
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def search_best_hyperparams(graph_edges, feat, device, exp_name=''):
    num_nodes = feat.shape[0]
    
    num_of_layers_grid = [1, 2]
    hidden_size_grid = [128, 256, 512] # [64, 128, 256] - for heterogen
    lr_grid = [0.0005, 0.001, 0.005, 0.01]
    gamma_grid = [0.9] # [0.7, 0.9] - for heterogen
    num_runs = 1

    train_edges, val_edges, _ = transductive_edge_split(graph_edges, num_nodes, num_val=0.2, num_test=0.0)

    best_val_metric = 0
    for num_of_layers in tqdm(num_of_layers_grid):
        for hidden_size in tqdm(hidden_size_grid):
            for lr in tqdm(lr_grid):
                for gamma in tqdm(gamma_grid):
                    metric_runs = []
                    for _ in range(num_runs):
                        model = GAT_PINE(num_of_layers=num_of_layers, 
                                        num_heads_per_layer=[1]*num_of_layers,
                                        num_features_per_layer=[feat.shape[1]] + [hidden_size]*num_of_layers, 
                                        add_skip_connection=True, bias=False,
                                        dropout=0.1, log_attention_weights=True)
                        model, val_metric_run = train(feat, train_edges, val_edges, model, lr, gamma, device, return_val_metric=True, 
                                                      earlystop_checkpoint_path=f'hyperparam_checkpoint_{exp_name}.pt', val_verbose=False)
                        metric_runs.append(val_metric_run)
                    val_metric = np.mean(metric_runs)
                    if val_metric > best_val_metric:
                        best_hyperparams = (num_of_layers, hidden_size, lr, gamma)
                        best_val_metric = val_metric
    
    best_hyperparams_res = {'num_of_layers': best_hyperparams[0], 
                            'hidden_size': best_hyperparams[1], 
                            'lr': best_hyperparams[2], 
                            'gamma': best_hyperparams[3]}
    
    return best_hyperparams_res
