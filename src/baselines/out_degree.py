import networkx as nx


def get_outdegree_nodes(G):
    outdegree = nx.out_degree_centrality(G)
    outdegree_sorted = {k: v for k, v in sorted(outdegree.items(), key=lambda item: item[1], reverse=True)}
    outdegree_nodes = list(outdegree_sorted.keys())
    return outdegree_nodes
