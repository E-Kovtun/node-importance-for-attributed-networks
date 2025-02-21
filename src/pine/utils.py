import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_node_mapping(G):
    id2node = {}
    node2id = {}
    id = 0
    for node in G.nodes():
        id2node[id] = node
        node2id[node] = id
        id += 1
    return id2node, node2id


def construct_adjacency_matrix(G, node2id, diag=True, directed=True):
    if diag:
        adj_matrix = np.eye(len(G.nodes))
    else:
        adj_matrix = np.zeros((len(G.nodes), len(G.nodes)))
    for (u, v) in G.edges():
        adj_matrix[node2id[u], node2id[v]] = 1
        if not directed:
            adj_matrix[node2id[v], node2id[u]] = 1
    return adj_matrix


def get_edge_index(G, node2id):
    graph_edges = []
    for (u, v) in G.edges():
        graph_edges.append((node2id[u], node2id[v]))
    return graph_edges


def get_edge_index_from_list(edge_list, node2id):
    graph_edges = []
    for (u, v) in edge_list:
        graph_edges.append((node2id[u], node2id[v]))
    return graph_edges


def get_feature_map(G, id2node, data_dict):
    feature_map = []
    for id in range(len(G.nodes())):
        node = id2node[id]
        feat = data_dict[node]['emb']
        feature_map.append(feat)
    feature_map = np.array(feature_map)
    return feature_map


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        return h_prime, attention
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        