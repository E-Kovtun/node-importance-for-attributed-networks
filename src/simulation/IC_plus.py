from copy import deepcopy
import random 
import networkx as nx
import numpy as np 
from tqdm import tqdm


def run_IC(G, S, tw, aw, alpha1=0.5, alpha2=0.5):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''

    T = deepcopy(S) # copy already selected nodes
    for u in T: 
        for v in G[u]: # check whether new node v is influenced by chosen node u
            # p = tw[(u, v)] * aw[(u, v)]
            p = (alpha1 * tw[(u, v)] + alpha2 * aw[(u, v)])
            if (v not in T) and (random.random() < p):
                T.append(v)
    return T

