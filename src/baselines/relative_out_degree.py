import numpy as np
import networkx as nx


def get_relative_outdegree_nodes(G, alpha=0.5):
    k = nx.out_degree_centrality(G)
    s = {node: np.sum([G[node][v]['weight'] for v in G.successors(node)]) for node in G.nodes()}
    generalized_outdegree = {node: k[node] * (s[node] / k[node])**alpha if k[node] > 0 else 0 for node in G.nodes()}
    return generalized_outdegree
