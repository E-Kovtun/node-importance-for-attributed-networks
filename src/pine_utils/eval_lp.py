import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm 
from src.backbone.lp_head import LinkPredHead


def test(feat, test_edges, model, device):
    link_head = LinkPredHead()

    feat = torch.tensor(feat, dtype=torch.float32).to(device)
    mp_test_edge_index = test_edges['edge_index'].to(device)
    superv_test_edge_index = test_edges['edge_label_index'].to(device)
    test_label = test_edges['edge_label'].to(device)

    model = model.to(device)
    model.eval()

    h_prime = model((feat, mp_test_edge_index))[0]
    lp_out = link_head(h_prime, superv_test_edge_index)

    test_accuracy = accuracy_score(test_label.detach().cpu().numpy(), (torch.sigmoid(lp_out) >= 0.5).detach().cpu().numpy())
    test_roc_auc = roc_auc_score(test_label.detach().cpu().numpy(), lp_out.detach().cpu().numpy())

    return test_accuracy, test_roc_auc
