import numpy as np
import networkx as nx


def BII(G,alpha,max_iter=26, tol=1.0e-6, weight='weight'): 
    N = nx.number_of_nodes(G)
    bo = N*tol
    W = stochastic_graph_uniform(G, weight=weight)
        
    biiv = dict.fromkeys(W, 1.0 / N )  #intial value 每个节点的初始值为1/N
     
    ind = dict(nx.in_degree_centrality(G))
    # power iteration: make up to max_iter iterations
     
    for iternum in range(max_iter):
        xlast = biiv
        biiv = dict.fromkeys(xlast.keys(), 0)
        for n in biiv:
            for nbr in W[n]: #W[n] 为节点n的邻居
                biiv[nbr] += alpha * xlast[n] * W[n][nbr][weight] #一个节点的重要性等于指向它的节点的重要性之和
            biiv[n] += ind[n]
        
        #对向量做二范数归一化，即每个元素平方和开根号
        bii_sum = np.linalg.norm(np.array(list(biiv.values())))
        for key in biiv:
             biiv[key] = biiv[key]/bii_sum

        err = sum([abs(biiv[n] - xlast[n]) for n in biiv])
        if err < bo:
            return biiv
            
    return biiv


def stochastic_graph_uniform(G, weight='weight'):
    #不考虑是否有边指向
    for u, v, d in G.edges(data=True):
        d[weight] = 1
    return G


def get_bii_nodes(G):
    bii_dict = BII(G.reverse(), alpha=1.0)
    bii_dict = dict(sorted(bii_dict.items(), key=lambda x: x[1], reverse=True))
    bii_nodes = list(bii_dict.keys())
    return bii_nodes
