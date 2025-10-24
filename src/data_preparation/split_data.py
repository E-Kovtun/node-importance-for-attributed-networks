import numpy as np
import random
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import torch


def transductive_edge_split(graph_edges, num_nodes, num_val=0.1, num_test=0.2, disjoint_train_ratio=0.3):
    transform = RandomLinkSplit(num_val=num_val, 
                                num_test=num_test,
                                is_undirected=False, 
                                add_negative_train_samples=True,
                                neg_sampling_ratio=1.0,
                                key = "edge_label", # supervision label
                                disjoint_train_ratio=disjoint_train_ratio, # disjoint mode if > 0
                                )    
    graph_data = Data(edge_index=torch.tensor(graph_edges), num_nodes=num_nodes)
    train_edges, val_edges, test_edges = transform(graph_data)
    return train_edges, val_edges, test_edges



# def split_edges(graph_edges, test_ratio=0.2, val_ratio=0.1):
#     eids = np.arange(graph_edges.shape[1])
#     eids = np.random.permutation(eids)
#     u, v = graph_edges

#     test_size = int(len(eids) * test_ratio)
#     val_size = int(len(eids) * val_ratio)

#     test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
#     val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
#     train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

#     train_edges = np.stack((train_pos_u, train_pos_v), axis=1).transpose()
#     val_edges = np.stack((val_pos_u, val_pos_v), axis=1).transpose()
#     test_edges = np.stack((test_pos_u, test_pos_v), axis=1).transpose()
#     return train_edges, val_edges, test_edges


# Split 2
# def split_edges(G, valid_ratio = 0.15, superv_train_ratio=0.3):
#     num_edges = len(G.edges())
#     valid_size = int(valid_ratio*num_edges)
#     superv_valid_edges = random.sample(list(G.edges()), valid_size)

#     mp_valid_G = G.copy()
#     mp_valid_G.remove_edges_from(superv_valid_edges)

#     print(f'MP valid graph num nodes {len(mp_valid_G.nodes())}, num_edges {len(mp_valid_G.edges())}')
#     superv_valid_edges = [(u, v) for (u, v) in superv_valid_edges if ((u in mp_valid_G.nodes()) and (v in mp_valid_G.nodes()))]
#     print(f'Number of superv valid edges', len(superv_valid_edges))

#     num_full_train_edges = len(mp_valid_G.edges())
#     superv_train_edges_size = int(superv_train_ratio * num_full_train_edges)
#     superv_train_edges = random.sample(list(mp_valid_G.edges()), superv_train_edges_size)

#     mp_train_G = mp_valid_G.copy()
#     mp_train_G.remove_edges_from(superv_train_edges)
#     print(f'MP train graph num nodes {len(mp_train_G.nodes())}, num_edges {len(mp_train_G.edges())}')
#     superv_train_edges = [(u, v) for (u, v) in superv_train_edges if ((u in mp_train_G.nodes()) and (v in mp_train_G.nodes()))]
#     print(f'Number of superv train edges', len(superv_train_edges))

#     train_edge_index = np.array(list(mp_train_G.edges())).T
#     superv_train_edge_index = np.array(superv_train_edges).T
#     val_edge_index = np.array(list(mp_valid_G.edges())).T
#     superv_val_edge_index = np.array(superv_valid_edges).T    

#     return train_edge_index, superv_train_edge_index, val_edge_index, superv_val_edge_index
