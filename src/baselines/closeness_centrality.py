import networkx as nx


def get_closeness_nodes(G):
    cc = nx.closeness_centrality(G.to_undirected())
    return cc
