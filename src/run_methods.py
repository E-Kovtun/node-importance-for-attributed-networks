import networkx as nx
import json
import os
import torch
import numpy as np
from src.utils import cos_sim, topology_weights, attribute_weights
from src.baselines.node_degree import get_degree_nodes
from src.baselines.out_degree import get_outdegree_nodes
from src.baselines.weighted_out_degree import get_weighted_outdegree_nodes
from src.baselines.relative_out_degree import get_relative_outdegree_nodes
from src.baselines.enrenew import get_entropy_nodes
from src.baselines.bii import get_bii_nodes
from src.baselines.pagerank import get_pagerank_nodes
from src.baselines.katz import get_katz_nodes
from src.baselines.dsli import get_dsli_nodes
from src.pine.pine import get_pine_nodes
from src.baselines.voterank_plus import get_voterank_simulation_result
from src.simulation.utils import get_simulation_result


def compare_methods(dataset_name, data_dict, vocab_size, num_starts, num_runs, device):
    G_undirected = nx.Graph()
    for u in data_dict:
        G_undirected.add_node(u)
        if len(data_dict[u]['out']) >= 1:
            for v in data_dict[u]['out']:
                if u != v:
                    G_undirected.add_edge(u, v)

    largest_cc = max(nx.connected_components(G_undirected), key=len)
    S = G_undirected.subgraph(largest_cc)
    list_cc = [c for c in sorted(nx.connected_components(G_undirected), key=len, reverse=True)]
    S = G_undirected.subgraph(list_cc[0])

    G = nx.DiGraph()
    for u in data_dict:
        if u in S.nodes():
            if len(data_dict[u]['out']) >= 1:
                for v in data_dict[u]['out']:
                    if u != v:
                        G.add_edge(u, v)
                        G[u][v]['weight'] = cos_sim(data_dict[u]['emb'], data_dict[v]['emb'])

    G_reversed = G.reverse()

    tw = topology_weights(G)
    aw = attribute_weights(G)

    degree_nodes = get_degree_nodes(G)
    outdegree_nodes = get_outdegree_nodes(G)
    weighted_outdegree_nodes = get_weighted_outdegree_nodes(G, data_dict)
    relative_outdegree_nodes = get_relative_outdegree_nodes(G, data_dict, alpha=0.3)
    entropy_nodes = get_entropy_nodes(G)
    bii_nodes = get_bii_nodes(G)
    pagerank_nodes = get_pagerank_nodes(G)
    katz_nodes = get_katz_nodes(G)
    dsli_nodes = get_dsli_nodes(G)
    pine_nodes = get_pine_nodes(G_reversed, data_dict, vocab_size, device)


    degree_sim_lt = get_simulation_result(G, degree_nodes, tw, aw, 'LT', num_starts, num_runs)
    outdegree_sim_lt = get_simulation_result(G, outdegree_nodes, tw, aw, 'LT', num_starts, num_runs)
    weighted_outdegree_sim_lt = get_simulation_result(G, weighted_outdegree_nodes, tw, aw, 'LT', num_starts, num_runs)
    relative_outdegree_sim_lt = get_simulation_result(G, relative_outdegree_nodes, tw, aw, 'LT', num_starts, num_runs)
    entropy_sim_lt = get_simulation_result(G, entropy_nodes, tw, aw, 'LT', num_starts, num_runs)
    voterank_sim_lt = get_voterank_simulation_result(G, tw, aw, 'LT', num_starts, num_runs)
    bii_sim_lt = get_simulation_result(G, bii_nodes, tw, aw, 'LT', num_starts, num_runs)
    pagerank_sim_lt = get_simulation_result(G, pagerank_nodes, tw, aw, 'LT', num_starts, num_runs)
    katz_sim_lt = get_simulation_result(G, katz_nodes, tw, aw, 'LT', num_starts, num_runs)
    dsli_sim_lt = get_simulation_result(G, dsli_nodes, tw, aw, 'LT', num_starts, num_runs)
    pine_sim_lt = get_simulation_result(G, pine_nodes, tw, aw, 'LT', num_starts, num_runs)

    degree_sim_ic = get_simulation_result(G, degree_nodes, tw, aw, 'IC', num_starts, num_runs)
    outdegree_sim_ic = get_simulation_result(G, outdegree_nodes, tw, aw, 'IC', num_starts, num_runs)
    weighted_outdegree_sim_ic = get_simulation_result(G, weighted_outdegree_nodes, tw, aw, 'IC', num_starts, num_runs)
    relative_outdegree_sim_ic = get_simulation_result(G, relative_outdegree_nodes, tw, aw, 'IC', num_starts, num_runs)
    entropy_sim_ic = get_simulation_result(G, entropy_nodes, tw, aw, 'IC', num_starts, num_runs)
    voterank_sim_ic = get_voterank_simulation_result(G, tw, aw, 'IC', num_starts, num_runs)
    bii_sim_ic = get_simulation_result(G, bii_nodes, tw, aw, 'IC', num_starts, num_runs)
    pagerank_sim_ic = get_simulation_result(G, pagerank_nodes, tw, aw, 'IC', num_starts, num_runs)
    katz_sim_ic = get_simulation_result(G, katz_nodes, tw, aw, 'IC', num_starts, num_runs)
    dsli_sim_ic = get_simulation_result(G, dsli_nodes, tw, aw, 'IC', num_starts, num_runs)
    pine_sim_ic = get_simulation_result(G, pine_nodes, tw, aw, 'IC', num_starts, num_runs)

    res = {'lt': {'degree': degree_sim_lt, 'outdegree': outdegree_sim_lt, 'weighted_outdegree': weighted_outdegree_sim_lt, 
                       'relative_outdegree': relative_outdegree_sim_lt, 'entropy': entropy_sim_lt, 
                       'pine': pine_sim_lt, 'voterank': voterank_sim_lt, 'bii': bii_sim_lt, 
                       'pagerank': pagerank_sim_lt, 'katz': katz_sim_lt, 'dsli': dsli_sim_lt}, 
                
                'ic': {'degree': degree_sim_ic, 'outdegree': outdegree_sim_ic, 'weighted_outdegree': weighted_outdegree_sim_ic, 
                       'relative_outdegree': relative_outdegree_sim_ic, 'entropy': entropy_sim_ic, 
                       'pine': pine_sim_ic, 'voterank': voterank_sim_ic, 'bii': bii_sim_ic, 
                       'pagerank': pagerank_sim_ic, 'katz': katz_sim_ic, 'dsli': dsli_sim_ic}}
    
    with open(f'./results/{dataset_name}_simres.json', 'w') as f:
        json.dump(res, f)

    return



if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs('results', exist_ok=True)

    with open('prepared_data/citeseer_dict.json', 'r') as f:
        data_dict = json.load(f)

    dataset_name = 'citeseer'

    vocab_size = len(next(iter(data_dict.values()))['emb'])

    num_starts = np.arange(5, 101, step=5)
    num_runs = 1000

    compare_methods(dataset_name, data_dict, vocab_size, num_starts, num_runs, device)
