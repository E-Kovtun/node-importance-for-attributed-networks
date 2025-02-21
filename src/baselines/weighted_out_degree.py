import numpy as np


def get_weighted_outdegree_nodes(G, data_dict):
    semantic_outdegree = {node: np.sum([G[node][v]['weight'] for v in data_dict[node]['out'] if node != v]) for node in G.nodes()}
    semantic_outdegree_sorted = {k: v for k, v in sorted(semantic_outdegree.items(), key=lambda item: item[1], reverse=True)}
    semantic_outdegree_nodes = list(semantic_outdegree_sorted.keys())
    return semantic_outdegree_nodes
