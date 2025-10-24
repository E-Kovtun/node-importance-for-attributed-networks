import networkx as nx


def get_pagerank_nodes(G):
    pagerank = nx.pagerank(G.reverse())
    return pagerank
