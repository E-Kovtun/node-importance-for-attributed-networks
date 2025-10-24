import numpy as np


def get_weighted_outdegree_nodes(G):
    semantic_outdegree = {node: np.sum([G[node][v]['weight'] for v in G.successors(node)]) for node in G.nodes()}
    return semantic_outdegree
