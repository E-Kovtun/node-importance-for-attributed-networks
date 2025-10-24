from copy import deepcopy
import random 
import networkx as nx
import numpy as np 
from tqdm import tqdm


def run_LT(G, S, tw, aw, alpha1=0.5, alpha2=0.5):
    '''
    Input: G -- networkx directed graph
    S -- initial seed set of nodes
    '''

    assert type(G) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(S) == list, 'Seed set S should be an instance of list'

    T = deepcopy(S)  # targeted set
    lv = dict()  # threshold for nodes
    for u in G:
        lv[u] = random.random()
    W = dict(zip(G.nodes(), [0]*len(G)))  # weighted number of activated in-neighbors

    Sj = deepcopy(S)  
    while len(Sj):  # while we have newly activated nodes
        Snew = []
        for u in Sj:
            for v in G[u]:  # In G，Sj u's out edge to v。
                if v not in T:
                    W[v] += (alpha1 * tw[(u, v)] + alpha2 * aw[(u, v)])
                    if W[v] >= lv[v]:  # if greater than threshold
                        Snew.append(v)
                        T.append(v)
        Sj = deepcopy(Snew)
        
    return T



# def run_LT(G, S, tw, aw, alpha1=0.5, alpha2=0.5, max_steps=100):
#     """
#     Run LTPlus diffusion on directed graph G.

#     Parameters:
#     - G: networkx.DiGraph
#     - S: list of initial seed nodes
#     - tw: dict {(u,v): topology weight}
#     - aw: dict {(u,v): attribute weight}
#     - alpha1: weight for topology influence
#     - alpha2: weight for attribute influence
#     - max_steps: max iterations to avoid infinite loops

#     Returns:
#     - activated: set of activated nodes at diffusion end
#     """

#     assert isinstance(G, nx.DiGraph), "G must be a networkx.DiGraph"
#     assert isinstance(S, list), "Seed set S must be a list"

#     # Step 1: Compute combined weights and normalize per node (incoming edges)
#     combined_weights = dict()
#     for v in G.nodes():
#         in_edges = list(G.in_edges(v))
#         total_weight = 0.0
#         for u, _ in in_edges:
#             w_topo = tw.get((u, v), 0)
#             w_attr = aw.get((u, v), 0)
#             combined = alpha1 * w_topo + alpha2 * w_attr
#             combined_weights[(u, v)] = combined
#             total_weight += combined
#         # Normalize if total_weight > 1 to ensure sum ≤ 1
#         if total_weight > 1.0:
#             for u, _ in in_edges:
#                 combined_weights[(u, v)] /= total_weight

#     # Step 2: Initialize thresholds and activated sets
#     thresholds = {node: random.random() for node in G.nodes()}
#     activated = set(S)
#     newly_activated = set(S)

#     # Step 3: Initialize accumulated influence per node
#     influence = {node: 0.0 for node in G.nodes()}

#     # Step 4: Diffusion process
#     step = 0
#     while newly_activated and step < max_steps:
#         step += 1
#         next_activated = set()
#         for u in newly_activated:
#             # For each neighbor v of u (outgoing edges)
#             for v in G.successors(u):
#                 if v not in activated:
#                     # Accumulate influence from u to v
#                     influence[v] += combined_weights.get((u, v), 0)
#                     # Check threshold
#                     if influence[v] >= thresholds[v]:
#                         next_activated.add(v)
#         activated.update(next_activated)
#         newly_activated = next_activated

#     return activated

