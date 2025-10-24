import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from src.utils import EarlyStopping
from torch.utils.data import DataLoader
from tqdm import tqdm 
from src.backbone.lp_head import LinkPredHead


def train(feat, train_edges, val_edges, model, lr, gamma, device, return_val_metric=False, n_epochs=10000, 
          earlystop_checkpoint_path="checkpoint.pt", val_verbose=True):
    early_stopping = EarlyStopping(patience=100, verbose=False, delta=1e-4, path=earlystop_checkpoint_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma) 
    link_head = LinkPredHead()
    loss_bce = nn.BCEWithLogitsLoss()

    feat = torch.tensor(feat, dtype=torch.float32).to(device)
    mp_train_edge_index = train_edges['edge_index'].to(device)
    superv_train_edge_index = train_edges['edge_label_index'].to(device)
    train_label = train_edges['edge_label'].to(device)

    mp_val_edge_index = val_edges['edge_index'].to(device)
    superv_val_edge_index = val_edges['edge_label_index'].to(device)
    val_label = val_edges['edge_label'].to(device)

    model = model.to(device)

    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()
        h_prime = model((feat, mp_train_edge_index))[0]
        lp_out = link_head(h_prime, superv_train_edge_index)
        loss = loss_bce(lp_out, train_label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        h_prime = model((feat, mp_val_edge_index))[0]
        lp_out = link_head(h_prime, superv_val_edge_index)
        val_loss = loss_bce(lp_out, val_label)
        val_metric = roc_auc_score(val_label.detach().cpu().numpy(), lp_out.detach().cpu().numpy())
        if val_verbose:
            if epoch % 10 == 0:
                print(f'Val Epoch: {epoch} -- Loss: {loss.item()} -- ROC AUC: {val_metric}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping of epoch {epoch}')
            break

    model.load_state_dict(torch.load(earlystop_checkpoint_path, weights_only=True))
        
    if return_val_metric:
        model.eval()
        h_prime = model((feat, mp_val_edge_index))[0]
        lp_out = link_head(h_prime, superv_val_edge_index)
        final_val_metric = roc_auc_score(val_label.detach().cpu().numpy(), lp_out.detach().cpu().numpy())
        return model, final_val_metric  

    return model



