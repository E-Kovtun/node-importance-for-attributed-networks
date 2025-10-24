from src.utils import cos_sim, topology_weights, attribute_weights, sort_nodes
import networkx as nx
from tqdm import tqdm
 

def construct_graph(feat, graph_edges):
    G = nx.DiGraph()
    for e in tqdm(range(graph_edges.shape[1])):
        u, v = graph_edges[0, e], graph_edges[1, e]
        G.add_edge(v, u)
        G[v][u]['weight'] = cos_sim(feat[v], feat[u])

    G.remove_edges_from(nx.selfloop_edges(G))
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    tw = topology_weights(G)
    aw = attribute_weights(G)

    return G, tw, aw
