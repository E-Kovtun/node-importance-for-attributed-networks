import networkx as nx 


def get_degree_nodes(G):
    degree = nx.degree_centrality(G)
    return degree
