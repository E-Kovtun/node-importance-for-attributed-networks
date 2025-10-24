import torch
import scipy.stats
import sklearn.metrics as sk
import numpy as np


def dcg(y_true, y_pred, top_k):
    with torch.no_grad():
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        _, pred_indices = y_pred.topk(k=top_k)
        gain = y_true.gather(-1, pred_indices)

        return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)

def ndcg(y_true, y_pred, top_k):
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    with torch.no_grad():
        return (dcg_score / idcg_score).item()

def spearman_sci(y_true, y_pred):
    y_true = y_true.reshape(-1).detach().cpu().numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().numpy()
    return scipy.stats.spearmanr(y_true, y_pred)[0]

def overlap(y_true, y_pred, topk):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape
    _, true_indices = y_true.topk(k=topk)
    _, pred_indices = y_pred.topk(k=topk)
    overlap_num = len(set(true_indices.tolist()) & set(pred_indices.tolist()))
    return overlap_num / topk

def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)

