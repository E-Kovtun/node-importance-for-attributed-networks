import pandas as pd
import numpy as np
import torch
from collections import Counter
import os
import json
from tqdm import tqdm
import math
import random
import pickle
import csv
import json
from src.data_preparation.prepare_heterogeneous_graph import prepare_heterogeneous_data, split_train_val_test, load_split_data
from src.metrics.supervised_metrics import get_rank_metrics, overlap
import argparse


def aggregation_pine(dataset_name, data_path, graph_data, semantic_data, split_data, num_split_idx, 
                  result_folder, pine_edge_types_folder, pine_edge_types_metrics, cross_num,
                  final_res_name, pine_thr, train_num):
    
    graph_data_path = data_path + '/' + dataset_name + '/' + graph_data
    semantic_data_path = data_path +'/' + dataset_name + '/' + semantic_data
    split_data_path = data_path + f'/{dataset_name}/datasets_split/' + split_data
    pine_node_scores_folder = os.path.join(result_folder, pine_edge_types_folder)
    pine_scores_file = os.path.join(result_folder, pine_edge_types_metrics)
    metrics_file = os.path.join(result_folder, final_res_name)
     
    res = pd.read_csv(pine_scores_file)
    
    graph_edges, feat, edge_types, _ = prepare_heterogeneous_data(graph_data_path, 
                                                                            semantic_data_path, 
                                                                            split_data_path, num_split_idx)
    
    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    
    test_pine_ndcg, test_pine_spearman, test_pine_overlap = [], [], []
    test_outdegree_ndcg, test_outdegree_spearman, test_outdegree_overlap = [], [], []
    for cross_id in range(cross_num):
        _, val_idx, test_idx, _, val_labels, test_labels, _, _ = split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx)
        val_node_labels = dict(zip(val_idx, val_labels))
        test_node_labels = dict(zip(test_idx, test_labels))
        print('Val size', len(val_idx))
        print('Test size', len(test_idx))
        
        candidate_type_val = {}
        for edge_type in tqdm(res['edge_type'].values):
            with open(os.path.join(pine_node_scores_folder, f'edge_type_{edge_type}.json'), 'r') as f:
                basic_pine_imp = json.load(f)  
                
            type_node_scores = {}
            for type_node in basic_pine_imp:
                if int(type_node) in val_idx:
                    if basic_pine_imp[type_node]:
                        type_node_scores[int(type_node)] = np.mean(basic_pine_imp[type_node])
                        
            pine_nodes = np.unique(list(type_node_scores.keys()))
            pine_label = torch.tensor([type_node_scores[node] for node in pine_nodes])
            
            type_subrgaph = graph_edges[:, edge_types==edge_type]
            outdegree_label = torch.tensor([np.sum(type_subrgaph[1, :]==node) for node in pine_nodes])
            gt_label = torch.tensor([val_node_labels[node] for node in pine_nodes])
                        
            if len(gt_label) > 100:
                pine_ndcg_score, pine_spearman_score = get_rank_metrics(pine_label, gt_label, 100, True)
                outdegree_ndcg_score, _ = get_rank_metrics(outdegree_label, gt_label, 100, True)
            else:
                continue

            # if (pine_ndcg_score > outdegree_ndcg_score) and (pine_spearman_score > pine_thr): 
            if pine_spearman_score > pine_thr:
                candidate_type_val[edge_type] = pine_ndcg_score
                
        sort_candidate_type_val = dict(sorted(candidate_type_val.items(), key=lambda item: item[1], reverse=True))
        n_scores = []
        for n in range(len(sort_candidate_type_val)):
            pine_val_scores = {node: 0 for node in val_idx}
            for edge_type in tqdm(list(sort_candidate_type_val.keys())[:n]):
                with open(os.path.join(pine_node_scores_folder, f'edge_type_{edge_type}.json'), 'r') as f:
                    basic_pine_imp = json.load(f)  
                    
                for type_node in basic_pine_imp:
                    if basic_pine_imp[type_node]:
                        if int(type_node) in val_idx:
                            pine_val_scores[int(type_node)] += np.mean(basic_pine_imp[type_node])
                            
            pine_nodes = np.unique(list(pine_val_scores.keys()))
            global_pine_label = torch.tensor([np.sum(pine_val_scores[node]) * np.sum(graph_edges[1, :]==node) for node in pine_nodes])
            outdegree_label = torch.tensor([np.sum(graph_edges[1, :]==node) for node in pine_nodes])
            full_gt_label = torch.tensor([val_node_labels[node] for node in pine_nodes])
            ndcg_score, _ = get_rank_metrics(global_pine_label, full_gt_label, 100, True) 
            n_scores.append(ndcg_score)
            
            best_n = np.argmax(n_scores)
            
        # Testing
        pine_test_scores = {node: 0 for node in test_idx}
        for edge_type in tqdm(list(sort_candidate_type_val.keys())[:best_n]):
            with open(os.path.join(pine_node_scores_folder, f'edge_type_{edge_type}.json'), 'r') as f:
                basic_pine_imp = json.load(f)  
                
            for type_node in basic_pine_imp:
                if basic_pine_imp[type_node]:
                    if int(type_node) in test_idx:
                        pine_test_scores[int(type_node)] += np.mean(basic_pine_imp[type_node])
        
        pine_nodes = np.unique(list(pine_test_scores.keys()))
        global_pine_label = torch.tensor([np.sum(pine_test_scores[node]) * np.sum(graph_edges[1, :]==node) for node in pine_nodes])
        outdegree_label = torch.tensor([np.sum(graph_edges[1, :]==node) for node in pine_nodes])
        full_gt_label = torch.tensor([test_node_labels[node] for node in pine_nodes])
        
        test_outdegree_ndcg_score, test_outdegree_spearman_score = get_rank_metrics(outdegree_label, full_gt_label, 100, True)
        test_outdegree_overlap_score = overlap(full_gt_label, outdegree_label, 100)
        test_outdegree_ndcg.append(test_outdegree_ndcg_score)
        test_outdegree_spearman.append(test_outdegree_spearman_score)
        test_outdegree_overlap.append(test_outdegree_overlap_score)
        
        test_pine_ndcg_score, test_pine_spearman_score = get_rank_metrics(global_pine_label, full_gt_label, 100, True)
        test_pine_overlap_score = overlap(full_gt_label, global_pine_label, 100)
        test_pine_ndcg.append(test_pine_ndcg_score)
        test_pine_spearman.append(test_pine_spearman_score)
        test_pine_overlap.append(test_pine_overlap_score)
        
    dict_metrics = {'pine': {'ndcg': test_pine_ndcg, 'mean_ndcg': np.mean(test_pine_ndcg), 'std_ndcg': np.std(test_pine_ndcg), 
                             'spearman': test_pine_spearman, 'mean_spearman': np.mean(test_pine_spearman), 'std_spearman': np.std(test_pine_spearman), 
                             'overlap': test_pine_overlap, 'mean_overlap': np.mean(test_pine_overlap), 'std_overlap': np.std(test_pine_overlap)}, 
                    'outdegree': {'ndcg': test_outdegree_ndcg, 'mean_ndcg': np.mean(test_outdegree_ndcg), 'std_ndcg': np.std(test_outdegree_ndcg), 
                                 'spearman': test_outdegree_spearman, 'mean_spearman': np.mean(test_outdegree_spearman), 'std_spearman': np.std(test_outdegree_spearman), 
                                 'overlap': test_outdegree_overlap, 'mean_overlap': np.mean(test_outdegree_overlap), 'std_overlap': np.std(test_outdegree_overlap)}}
    
    with open(metrics_file, 'w') as f:
        json.dump(dict_metrics, f)
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
    parser.add_argument("--pine_edge_types_folder", type=str, default='FB15K_pine_importances_base',
                        help="Folder with PINE scores for subgraphs of different edge types") 
    parser.add_argument("--pine_edge_types_metrics", type=str, default='FB15K_res_base.csv',
                        help="File with metrics for different edge types")    
    parser.add_argument("--final_res_name", type=str, default='FB15K_final_supervised_res.json',
                        help="Name of json file with final supervised metrics")   
    parser.add_argument("--cross_num", type=int, default=5,
                        help="Number of cross validation folds")   
    parser.add_argument("--pine_thr", type=float, default=0.0,
                        help="Threshold on Spearman correlation of PINE scores for one edge type and global ground truth scores")   
    parser.add_argument("--train_num", type=int, default=8,
                        help="Number of folds to include into train set")      
    args = parser.parse_args()
    
    
    aggregation_pine(args.dataset_name, args.data_path, args.graph_data, args.semantic_data, args.split_data, args.num_split_idx, 
                     args.result_folder, args.pine_edge_types_folder, args.pine_edge_types_metrics,
                     args.cross_num, args.final_res_name, args.pine_thr, args.train_num)
    
    