import networkx as nx


def get_pagerank_nodes(G):
    pagerank = nx.pagerank(G.reverse())
    pagerank_sorted = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1], reverse=True)}
    pagerank_nodes = list(pagerank_sorted.keys())
    return pagerank_nodes
