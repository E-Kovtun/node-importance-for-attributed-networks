import networkx as nx


def get_voterank_nodes(G):
    voterank = nx.voterank(G.to_undirected())
    voterank = {id: len(voterank)-i for i, id in enumerate(voterank)}
    return voterank
