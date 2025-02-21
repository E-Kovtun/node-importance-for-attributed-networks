import torch
import torch.nn as nn
import random
import numpy as np
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

from src.pine.utils import get_node_mapping, construct_adjacency_matrix, get_edge_index, get_edge_index_from_list, get_feature_map
from src.pine.utils import GraphAttentionLayer
from src.pine.utils import EarlyStopping


class GAT_link(nn.Module):
    def __init__(self, emb_dim, hidden_size=64, out_features=32):
        super(GAT_link, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.out_features = out_features

        self.in_linear = nn.Linear(self.emb_dim, self.hidden_size) 
        self.attention_layer1 = GraphAttentionLayer(self.hidden_size, self.hidden_size)
        self.attention_layer2 = GraphAttentionLayer(self.hidden_size, self.out_features)
        # self.attention_layer3 = GraphAttentionLayer(self.out_features, self.out_features)

    def forward(self, x, adj):
        h = self.in_linear(x)
        h_prime1, attn1 = self.attention_layer1(h, adj)
        h_prime2, attn2 = self.attention_layer2(h_prime1, adj)
        # h_prime3, attn3 = self.attention_layer3(h_prime2, adj)
        return h_prime2,  torch.mean(torch.stack([attn1, attn2], dim=0), dim=0)
    

class LinkPredHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        return torch.sum(x_src * x_dst, dim=1)
    

def lp_negative_sampling(edge_index, num_nodes):
    neg_edge_index = negative_sampling(edge_index, num_nodes, force_undirected=True)
    return neg_edge_index


def get_pine_nodes(G_reversed, data_dict, vocab_size, device, n_epochs=500, valid_ratio = 0.15, superv_train_ratio=0.3):
    id2node, node2id = get_node_mapping(G_reversed)
    adj_matrix = construct_adjacency_matrix(G_reversed, node2id, diag=False, directed=True)
    graph_edges = get_edge_index(G_reversed, node2id)
    feature_map = get_feature_map(G_reversed, id2node, data_dict)

    features = torch.tensor(feature_map, dtype=torch.float32).to(device)
    adj = torch.tensor(adj_matrix, dtype=torch.int64).to(device)
    edge_index = torch.tensor(np.array(graph_edges).T, dtype=torch.int64)

    print(f'Initial graph num nodes {len(G_reversed.nodes())}, num_edges {len(G_reversed.edges())}')
    num_edges = len(G_reversed.edges())
    valid_size = int(valid_ratio*num_edges)
    superv_valid_edges = random.sample(list(G_reversed.edges()), valid_size)

    mp_valid_G = G_reversed.copy()
    mp_valid_G.remove_edges_from(superv_valid_edges)

    print(f'MP valid graph num nodes {len(mp_valid_G.nodes())}, num_edges {len(mp_valid_G.edges())}')
    superv_valid_edges = [(u, v) for (u, v) in superv_valid_edges if ((u in mp_valid_G.nodes()) and (v in mp_valid_G.nodes()))]
    print(f'Number of superv valid edges', len(superv_valid_edges))

    num_full_train_edges = len(mp_valid_G.edges())
    superv_train_edges_size = int(superv_train_ratio * num_full_train_edges)
    superv_train_edges = random.sample(list(mp_valid_G.edges()), superv_train_edges_size)


    mp_train_G = mp_valid_G.copy()
    mp_train_G.remove_edges_from(superv_train_edges)
    print(f'MP train graph num nodes {len(mp_train_G.nodes())}, num_edges {len(mp_train_G.edges())}')
    superv_train_edges = [(u, v) for (u, v) in superv_train_edges if ((u in mp_train_G.nodes()) and (v in mp_train_G.nodes()))]
    print(f'Number of superv train edges', len(superv_train_edges))


    train_id2node, train_node2id = get_node_mapping(mp_train_G)
    train_feature_map = get_feature_map(mp_train_G, train_id2node, data_dict)
    train_features = torch.tensor(train_feature_map, dtype=torch.float32).to(device)
    train_adj_matrix = construct_adjacency_matrix(mp_train_G, train_node2id, diag=False, directed=True) # ft
    train_adj = torch.tensor(train_adj_matrix, dtype=torch.int64).to(device)
    train_graph_edges = get_edge_index(mp_train_G, train_node2id)
    train_edge_index = torch.tensor(np.array(train_graph_edges).T, dtype=torch.int64)
    superv_train_edges = get_edge_index_from_list(superv_train_edges, train_node2id)
    superv_train_edge_index = torch.tensor(np.array(superv_train_edges).T, dtype=torch.int64)

    valid_id2node, valid_node2id = get_node_mapping(mp_valid_G)
    valid_feature_map = get_feature_map(mp_valid_G, valid_id2node, data_dict)
    valid_features = torch.tensor(valid_feature_map, dtype=torch.float32).to(device)
    valid_adj_matrix = construct_adjacency_matrix(mp_valid_G, valid_node2id, diag=False, directed=True)
    valid_adj = torch.tensor(valid_adj_matrix, dtype=torch.int64).to(device)
    valid_graph_edges = get_edge_index(mp_valid_G, valid_node2id)
    valid_edge_index = torch.tensor(np.array(valid_graph_edges).T, dtype=torch.int64)
    superv_valid_edges = get_edge_index_from_list(superv_valid_edges, valid_node2id)
    superv_valid_edge_index = torch.tensor(np.array(superv_valid_edges).T, dtype=torch.int64)

    model = GAT_link(emb_dim=vocab_size).to(device)
    link_head = LinkPredHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9) 
    loss_bce = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=10, verbose=False) # path='results/pine_checkpoint.pt'

    train_epoch = []
    valid_epoch = []
    train_epoch_loss = []
    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()
        h_prime, _ = model(train_features, train_adj)
        pos_out = link_head(h_prime, superv_train_edge_index.to(device))
        neg_edge_index = lp_negative_sampling(superv_train_edge_index, train_features.shape[0])
        neg_out = link_head(h_prime, neg_edge_index.to(device))

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]).to(device)
        loss = loss_bce(out, gt)
        train_metric = roc_auc_score(gt.detach().cpu().numpy(), out.detach().cpu().numpy())
        train_epoch.append(train_metric)
        # if epoch % 10 == 0:
        #     print(f'Train Epoch: {epoch} -- Loss: {loss.item()} -- ROC AUC: {train_metric}')
        train_epoch_loss.append(loss.item())
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        model.eval()
        h_prime, _ = model(valid_features, valid_adj)
        pos_out = link_head(h_prime, superv_valid_edge_index.to(device))
        neg_edge_index = lp_negative_sampling(superv_valid_edge_index, valid_features.shape[0])
        neg_out = link_head(h_prime, neg_edge_index.to(device))
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]).to(device)
        valid_loss = loss_bce(out, gt).item()
        valid_metric = roc_auc_score(gt.detach().cpu().numpy(), out.detach().cpu().numpy())
        valid_epoch.append(valid_metric)
        if epoch % 10 == 0:
            print(f'Valid Epoch: {epoch} -- Loss: {loss.item()} -- ROC AUC: {valid_metric}')

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping of epoch {epoch}')
            break
    
    model.eval()
    h_prime, attn = model(features, adj)
    h_prime = h_prime.detach().cpu().numpy()
    attn = attn.detach().cpu().numpy()
    attn_degree = {id: np.sum(attn[:, id]) for id in range(len(G_reversed.nodes()))}
    pine_dict = {id2node[id]: attn_degree[id] for id in attn_degree}
    pine_dict = dict(sorted(pine_dict.items(), key=lambda x: x[1], reverse=True))
    pine_nodes = list(pine_dict.keys())

    return pine_nodes