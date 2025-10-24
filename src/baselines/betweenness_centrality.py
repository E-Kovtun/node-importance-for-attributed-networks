import networkx as nx


def get_betweenness_nodes(G):
    bc = nx.betweenness_centrality(G.to_undirected())
    return bc
