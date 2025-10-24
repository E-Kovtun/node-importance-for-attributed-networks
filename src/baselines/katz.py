import networkx as nx

def get_katz_nodes(G):
    katz = nx.katz_centrality(G.reverse())
    return katz
