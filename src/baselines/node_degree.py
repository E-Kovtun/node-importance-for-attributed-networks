import networkx as nx 


def get_degree_nodes(G):
    degree = nx.degree_centrality(G)
    degree_sorted = {k: v for k, v in sorted(degree.items(), key=lambda item: item[1], reverse=True)}
    degree_nodes = list(degree_sorted.keys())
    return degree_nodes
