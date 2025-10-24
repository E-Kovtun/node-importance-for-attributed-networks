from collections import Counter
import numpy as np
import torch
from src.data_preparation.prepare_heterogeneous_graph import prepare_heterogeneous_data
from src.pine_utils.get_best_hyperparams import search_best_hyperparams
from src.data_preparation.split_data import transductive_edge_split
from src.backbone.gat_model import GAT_PINE
from src.pine_utils.train_lp import train
from src.pine_utils.eval_lp import test
from src.pine import calculate_pine_scores
from src.metrics.supervised_metrics import get_rank_metrics, overlap
import csv
import os
import json
import argparse


def heterogeneous_pine_exps(dataset_name, exp_name, data_path, graph_data, semantic_data, split_data, 
                            num_split_idx, result_folder, device, thr_size=1e3, num_runs=10):
    
    graph_data_path = data_path + '/' + dataset_name + '/' + graph_data
    semantic_data_path = data_path +'/' + dataset_name + '/' + semantic_data
    split_data_path = data_path + f'/{dataset_name}/datasets_split/' + split_data
    os.makedirs(result_folder, exist_ok=True)
    
    graph_edges, feat, edge_types, node_labels = prepare_heterogeneous_data(graph_data_path, 
                                                                            semantic_data_path, 
                                                                            split_data_path, num_split_idx)
    
    edge_type_sizes = dict(sorted(Counter(edge_types).items(), key=lambda item: item[1], reverse=True))
    write_header = True
    for edge_type in edge_type_sizes:
        if edge_type_sizes[edge_type] >= thr_size:
            subgraph_edges = graph_edges[:, edge_types==edge_type]
            subnodes = np.unique(subgraph_edges)
            subfeat = feat[subnodes, :]
            num_nodes = subfeat.shape[0]
            
            origin2new_dict = {subnodes[i]: i for i in range(len(subnodes))}
            new2origin_dict = {value: key for key, value in origin2new_dict.items()}
            f = lambda x: origin2new_dict[x]
            subgraph_edges = np.vectorize(f)(subgraph_edges)
            
            gt_labels = torch.tensor([node_labels[node] if node in node_labels else -1 for node in subnodes], dtype=torch.float32)
            gt_mask = gt_labels != -1
            gt_labels = gt_labels[gt_mask]
                    
            if len(gt_labels) > 100:
                best_hyperparams_res = search_best_hyperparams(subgraph_edges, subfeat, device, exp_name=exp_name)
                
                val_roc_auc_metrics, test_acc_metrics, test_roc_auc_metrics = [], [], []
                ndcg_metrics, sperman_metrics, overlap_metrics = [], [], []
                train_edges, val_edges, test_edges = transductive_edge_split(subgraph_edges, num_nodes, num_val=0.15, num_test=0.15)
                pine_scores = {node: [] for node in subnodes}
                for _ in range(num_runs):        
                    num_of_layers, hidden_size, lr, gamma = best_hyperparams_res['num_of_layers'], best_hyperparams_res['hidden_size'], best_hyperparams_res['lr'], best_hyperparams_res['gamma']                        
                    model = GAT_PINE(num_of_layers=num_of_layers, 
                                    num_heads_per_layer=[1]*num_of_layers,
                                    num_features_per_layer=[subfeat.shape[1]] + [hidden_size]*num_of_layers, 
                                    add_skip_connection=True, bias=False,
                                    dropout=0.1, log_attention_weights=True)
                    model, val_metric_run = train(subfeat, train_edges, val_edges, model, lr, gamma, device, return_val_metric=True, 
                                                earlystop_checkpoint_path=f'pine_checkpoint_{exp_name}.pt', val_verbose=False)
                    
                    attention_is_valid, pine_imp = calculate_pine_scores(model, subfeat, subgraph_edges, device)
                    if (val_metric_run > 0.5) and attention_is_valid: 
                        val_roc_auc_metrics.append(val_metric_run)
                        test_accuracy, test_roc_auc = test(subfeat, test_edges, model, device)
                        test_acc_metrics.append(test_accuracy)
                        test_roc_auc_metrics.append(test_roc_auc)                
                    
                        for id in range(len(pine_imp)):
                            node = new2origin_dict[id]
                            pine_scores[node].append(pine_imp[id])
                            
                        pine_labels = torch.tensor(pine_imp, dtype=torch.float32)
                        pine_labels = pine_labels[gt_mask]
                        
                        ndcg_score, spearman_score = get_rank_metrics(pine_labels, gt_labels, 100, True)
                        overlap_score = overlap(gt_labels, pine_labels, 100)
                        ndcg_metrics.append(ndcg_score)
                        sperman_metrics.append(spearman_score)
                        overlap_metrics.append(overlap_score)
                        
                degree_labels = torch.tensor([np.sum(subgraph_edges[1, :]==origin2new_dict[node]) for node in subnodes], dtype=torch.float32)
                degree_labels = degree_labels[gt_mask]
                
                degree_ndcg, degree_spearman = get_rank_metrics(degree_labels, gt_labels, 100, True)
                degree_overlap = overlap(gt_labels, degree_labels, 100)
                    
                edge_type_res = {**best_hyperparams_res, 
                                **{"val_roc_auc": val_roc_auc_metrics, "test_accuracy": test_acc_metrics, "test_roc_auc": test_roc_auc_metrics}, 
                                **{"pine_ndcg": ndcg_metrics, "pine_spearman": sperman_metrics, "pine_overlap": overlap_metrics}, 
                                **{"degree_ndcg": degree_ndcg, "degree_spearman": degree_spearman, "degree_overlap": degree_overlap}} 
                                
                result_file = f"{dataset_name}_res_{exp_name}.csv"
                row = {'edge_type': edge_type}
                row.update(edge_type_res)
                with open(os.path.join(result_folder, result_file), mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['edge_type'] + list(edge_type_res.keys()))
                    
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row)
                    
                pine_imp_folder = os.path.join(result_folder, f"{dataset_name}_pine_importances_{exp_name}")
                os.makedirs(pine_imp_folder, exist_ok=True)
                
                pine_file = f"edge_type_{edge_type}.json"
                pine_scores = {int(key): list(map(lambda x: float(x), value)) for key, value in pine_scores.items()}
                with open(os.path.join(pine_imp_folder, pine_file), 'w') as f:
                    json.dump(pine_scores, f)
    return 
                         
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heterogeneous PINE')
    
    parser.add_argument("--dataset_name", type=str, default='FB15K',
                        help="Name of the heterogeneous dataset to run experiments on")
    parser.add_argument("--data_path", type=str, default='./heterogeneous_data',
                        help="Name of the folder with heterogeneous datasets")  
    parser.add_argument("--exp_name", type=str, default='base',
                        help="Name of the current experiment")        
    parser.add_argument("--graph_data", type=str, default='fb15k_rel.pk',
                        help="Name of the file that contains graph data and structural features")          
    parser.add_argument("--semantic_data", type=str, default='fb_lang.pk',
                        help="Name of the file that contains semantic features")    
    parser.add_argument("--split_data", type=str, default='idx_1000',
                        help="Folder name with data on node IDs and their labels")    
    parser.add_argument("--num_split_idx", type=int, default=1000,
                        help="Number of data splits in folder 'split_data'")      
    parser.add_argument("--result_folder", type=str, default='./heterogeneous_results',
                        help="Folder name for saving results'")        
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device for PINE training'") 
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of times to launch PINE on each edge type of the graph") 
    
    args = parser.parse_args()
    
    heterogeneous_pine_exps(args.dataset_name, args.exp_name, args.data_path, 
                            args.graph_data, args.semantic_data, args.split_data, 
                            args.num_split_idx, args.result_folder, args.device, args.num_runs)
    
    