import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from src.data_preparation.split_data import transductive_edge_split
from src.backbone.gat_model import GAT_PINE
from src.backbone.lp_head import LinkPredHead
from src.pine_utils.train_lp import train
from src.pine_utils.eval_lp import test
from src.pine_utils.get_best_hyperparams import search_best_hyperparams
import time


def get_pine_nodes(exp_name, feat, graph_edges, device, measure_time=False, num_runs=5):
    best_hyperparams_res = search_best_hyperparams(graph_edges, feat, device, exp_name=exp_name)
    graph_nodes = np.unique(graph_edges)
    num_nodes = len(graph_nodes)
    
    val_roc_auc_metrics, test_acc_metrics, test_roc_auc_metrics = [], [], []
    train_edges, val_edges, test_edges = transductive_edge_split(graph_edges, num_nodes, num_val=0.15, num_test=0.15)
    pine_scores = {node: [] for node in graph_nodes}
    running_times = []
    for _ in range(num_runs):        
        start_time = time.time()
        num_of_layers, hidden_size, lr, gamma = best_hyperparams_res['num_of_layers'], best_hyperparams_res['hidden_size'], best_hyperparams_res['lr'], best_hyperparams_res['gamma']                        
        model = GAT_PINE(num_of_layers=num_of_layers, 
                        num_heads_per_layer=[1]*num_of_layers,
                        num_features_per_layer=[feat.shape[1]] + [hidden_size]*num_of_layers, 
                        add_skip_connection=True, bias=False,
                        dropout=0.1, log_attention_weights=True)
        model, val_metric_run = train(feat, train_edges, val_edges, model, lr, gamma, device, return_val_metric=True, 
                                        earlystop_checkpoint_path=f'pine_checkpoint_{exp_name}.pt', val_verbose=False)
        
        attention_is_valid, pine_imp = calculate_pine_scores(model, feat, graph_edges, device)
        end_time = time.time()
        if (val_metric_run > 0.5) and attention_is_valid: 
            running_times.append(end_time - start_time)
            val_roc_auc_metrics.append(val_metric_run)
            test_accuracy, test_roc_auc = test(feat, test_edges, model, device)
            test_acc_metrics.append(test_accuracy)
            test_roc_auc_metrics.append(test_roc_auc)        
    
            for id in range(len(pine_imp)):
                pine_scores[id].append(pine_imp[id])
    
    model_pine_res = {**best_hyperparams_res, 
                      **{"val_roc_auc": [np.mean(val_roc_auc_metrics), np.std(val_roc_auc_metrics)], 
                         "test_accuracy": [np.mean(test_acc_metrics), np.std(test_acc_metrics)], 
                         "rest_roc_auc": [np.mean(test_roc_auc_metrics), np.std(test_roc_auc_metrics)]}}                    
                             
    pine_dict = {id: np.mean(pine_scores[id]) for id in range(len(pine_scores))}
    if measure_time:
        delta_time = np.mean(running_times)
        return model_pine_res, pine_dict, delta_time
    else:
        return model_pine_res, pine_dict


def calculate_pine_scores(model, feat, graph_edges, device):
    model.eval()
    feat = torch.tensor(feat, dtype=torch.float32).to(device)
    graph_edges = torch.tensor(graph_edges, dtype=torch.int64).to(device)   
    _ = model((feat, graph_edges))[0]
    attention_weights = model.gat_net[0].attention_weights[:, 0, 0]
    pine_importances = torch.zeros(feat.shape[0], dtype=torch.float32, device=attention_weights.device)
    pine_importances.scatter_add_(0, graph_edges[1, :], attention_weights)
    pine_importances = pine_importances.detach().cpu().numpy()
    attention_is_valid = torch.sum(attention_weights[graph_edges[0, :]==graph_edges[0, 0]]).item() == 1
    return attention_is_valid, pine_importances