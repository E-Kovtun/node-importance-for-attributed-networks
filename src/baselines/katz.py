import networkx as nx

def get_katz_nodes(G):
    katz = nx.katz_centrality(G.reverse())
    katz_sorted = {k: v for k, v in sorted(katz.items(), key=lambda item: item[1], reverse=True)}
    katz_nodes = list(katz_sorted.keys())
    return katz_nodes
