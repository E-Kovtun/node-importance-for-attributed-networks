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
