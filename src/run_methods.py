import networkx as nx
import json
import os
import torch
import numpy as np
from src.data_preparation.prepare_graph import get_graph
from src.utils import sort_nodes
from src.baselines.node_degree import get_degree_nodes
from src.baselines.out_degree import get_outdegree_nodes
from src.baselines.weighted_out_degree import get_weighted_outdegree_nodes
from src.baselines.relative_out_degree import get_relative_outdegree_nodes
from src.baselines.pagerank import get_pagerank_nodes
from src.baselines.katz import get_katz_nodes
from src.baselines.closeness_centrality import get_closeness_nodes
from src.baselines.betweenness_centrality import get_betweenness_nodes
from src.baselines.voterank import get_voterank_nodes
from src.pine import get_pine_nodes
from src.simulation.utils import get_simulation_result
from src.data_preparation.graph_construction import construct_graph
from src.baselines.enrenew import get_entropy_nodes
from tqdm import tqdm
import time
import csv
import argparse


def compare_methods(datasets, measure_names, propagation_model_names, 
                    res_folder, device, node_ratio, num_runs):
    
    for dataset_name in datasets:
        feat, graph_edges = get_graph(dataset_name)
        num_starts = [np.round(feat.shape[0]*node_ratio).astype(np.int64)]

        G, tw, aw = construct_graph(feat, graph_edges)

        os.makedirs(res_folder, exist_ok=True)
        res_path = os.path.join(res_folder, f'{dataset_name}_{num_runs}.csv')
        columns = ['measure_name']
        for prop_name in propagation_model_names:
            columns += [f'{prop_name}_mean', f'{prop_name}_std']
        columns = columns + ['time']
        with open(res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            
        for measure_name in  measure_names:
            row2write = [measure_name]
            if measure_name == 'degree':
                start_time = time.time()
                imp_dict = get_degree_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time
            elif measure_name == 'out-degree':
                start_time = time.time()
                imp_dict = get_outdegree_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time       
            elif measure_name == 'weighted':
                start_time = time.time()
                imp_dict = get_weighted_outdegree_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time              
            elif measure_name == 'relative':
                start_time = time.time()
                imp_dict = get_relative_outdegree_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time   
            elif measure_name == 'pagerank':
                start_time = time.time()
                imp_dict = get_pagerank_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time             
            elif measure_name == 'katz':
                start_time = time.time()
                imp_dict = get_katz_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time                 
            elif measure_name == 'closeness':
                start_time = time.time()
                imp_dict = get_closeness_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time  
            elif measure_name == 'betweenness':
                start_time = time.time()
                imp_dict = get_betweenness_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time   
            elif measure_name == 'pine': 
                model_pine_res, imp_dict, delta_time = get_pine_nodes(dataset_name, feat, graph_edges, device, measure_time=True)
                with open(os.path.join(res_folder, f'pine_{dataset_name}_model_params.json'), 'w') as f:
                    json.dump(model_pine_res, f)
            elif measure_name == 'voterank':
                start_time = time.time()
                imp_dict = get_voterank_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time               
            elif measure_name == 'entropy_dir':    
                start_time = time.time()
                imp_dict = get_entropy_nodes(G)
                end_time = time.time()
                delta_time = end_time - start_time        
            else:
                print('Unknown measure name!')     
                
            imp_nodes = sort_nodes(imp_dict)
            
            for prop_name in propagation_model_names:
                if prop_name == 'LT+':
                    sim_lt = get_simulation_result(G, imp_nodes, tw, aw, 'LT', num_starts, num_runs)
                    sim_lt_mean, sim_lt_std = sim_lt[0][0], sim_lt[1][0]
                    row2write.extend([sim_lt_mean, sim_lt_std])
                elif prop_name == 'IC+':
                    sim_ic = get_simulation_result(G, imp_nodes, tw, aw, 'IC', num_starts, num_runs)
                    sim_ic_mean, sim_ic_std = sim_ic[0][0], sim_ic[1][0]
                    row2write.extend([sim_ic_mean, sim_ic_std])
                elif prop_name == 'SIR':
                    sim_sir = get_simulation_result(G, imp_nodes, tw, aw, 'SIR', num_starts, num_runs)
                    sim_sir_mean, sim_sir_std = sim_sir[0][0], sim_sir[1][0]
                    row2write.extend([sim_sir_mean, sim_sir_std])
                else:
                    print('Unknown propagation model')
            row2write.append(delta_time)
            with open(res_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row2write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINE')
    parser.add_argument("--dataset_names", nargs='+', default=['cora', 'citeseer', 'pubmed', 'wiki-cs', 'hepth', 'ogbn-arxiv', 'dblp'],
                        help="List of dataset names under study. \
                        Possible options: 'cora', 'citeseer', 'pubmed', 'wiki-cs', 'hepth', 'ogbn-arxiv', 'dblp'")
    parser.add_argument("--measure_names", nargs='+', default=['pine', 'degree', 'out-degree', 'weighted', 'relative', 'pagerank', 'voterank', 'katz', 'closeness', 'betweenness', 'entropy_dir'],
                        help="List of measure names to calculate node importance. \
                        Note that some measures are infeasible for calculation on large graphs. \
                        Possible options: 'pine', 'degree', 'out-degree', 'weighted', 'relative', 'pagerank', 'voterank', 'katz', \
                        'closeness', 'betweenness', 'entropy_dir'")
    parser.add_argument("--propagation_model_names", nargs='+', default=['LT+', 'IC+', 'SIR'],
                        help="List of propagation model names to evaluate performance of importance measures. \
                        Possible options: 'LT+', 'IC+', 'SIR'")
    parser.add_argument("--res_folder", type=str, default='./simulation_results',
                        help="Folder name for result saving")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to train PINE on")   
    parser.add_argument("--node_ratio", type=float, default=0.1,
                        help="Part of the nodes from which information dissemination process starts. \
                            0.1 means that 10\% of nodes with the greatest importance scores are initialized as active")      
    parser.add_argument("--num_runs", type=int, default=1000,
                        help="Number of Monte-Carlo simulation runs for propagation models: LT+, IC+, and SIR")          
    
    args = parser.parse_args()
    compare_methods(args.dataset_names, args.measure_names, args.propagation_model_names, 
                    args.res_folder, args.device, args.node_ratio, args.num_runs)


