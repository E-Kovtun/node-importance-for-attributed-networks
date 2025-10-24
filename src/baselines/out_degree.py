import networkx as nx


def get_outdegree_nodes(G):
    outdegree = nx.out_degree_centrality(G)
    return outdegree
