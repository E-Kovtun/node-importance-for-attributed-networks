import numpy as np
import torch

def topology_weights(G):
    """"
    Calculate topology weights for edges
    G: directed weighted networkx graph
    return: dict with edges as keys and topology weights as values
    """
    tw = dict()
    for u in G:
        in_edges = G.in_edges(u)
        indegree = len(in_edges)
        for (v, _) in in_edges:
            tw[(v, u)] = 1 / indegree
    return tw


def attribute_weights(G):
    """"
    Calculate attribute weights for edges and normalize them
    G: directed weighted networkx graph
    return: dict with edges as keys and attribute weights as values
    """
    aw = dict()
    for u in G:
        in_edges = G.in_edges(u)
        total = 0
        for (v, _) in in_edges:
            total += np.exp(G[v][u]['weight'])
        for (v, _) in in_edges:
            aw[(v, u)] = np.exp(G[v][u]['weight']) / total
    return aw


def cos_sim(emb1, emb2):
    if np.linalg.norm(emb1) * np.linalg.norm(emb2) > 0:
        return np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    else:
        return 0
    

def sort_nodes(imp_dict):
    imp_sorted = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}
    imp_nodes = list(imp_sorted.keys())
    return imp_nodes


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
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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

