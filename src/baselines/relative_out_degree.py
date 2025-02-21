import numpy as np
import networkx as nx


def get_relative_outdegree_nodes(G, data_dict, alpha=0.5):
    k = nx.out_degree_centrality(G)
    s = {node: np.sum([G[node][v]['weight'] for v in data_dict[node]['out'] if node != v]) for node in G.nodes()}
    generalized_outdegree = {node: k[node] * (s[node] / k[node])**alpha if k[node] > 0 else 0 for node in G.nodes()}
    generalized_outdegree_sorted = {k: v for k, v in sorted(generalized_outdegree.items(), key=lambda item: item[1], reverse=True)}
    generalized_outdegree_nodes = list(generalized_outdegree_sorted.keys())
    return generalized_outdegree_nodes
