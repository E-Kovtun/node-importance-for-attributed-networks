import pickle 
import torch
import numpy as np
import random 


def load_split_data(split_data_path, num_split_idx):
    dataset_spilt = []
    labels_idx = []
    for i in range(num_split_idx):
        with open(f"{split_data_path}/split_dataset_idx{i}.pkl", "rb") as input:
            data = pickle.load(input)
        dataset_spilt.append(data['idx'])
        labels_idx.append(data['labels'])  
    return dataset_spilt, labels_idx


def prepare_heterogeneous_data(graph_data_path, semantic_data_path, split_data_path, num_split_idx):
    with open(graph_data_path, 'rb') as f:
        data = pickle.load(f)
        
    graph_edges = torch.stack(data['edges'], dim=0).numpy()
    with open(semantic_data_path, 'rb') as f:
        feat = pickle.load(f)
    edge_types = data['edge_types'].numpy()
    
    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    dataset_spilt = np.concatenate(dataset_spilt, axis=0)
    labels_idx = np.concatenate(labels_idx, axis=0)
    node_labels = dict(zip(dataset_spilt, labels_idx))
    return graph_edges, feat, edge_types, node_labels
    
    
def split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx):
    dataset_index = [i for i in range(num_split_idx)]
    train_labels = np.array([], dtype=np.float32)
    train_idx = np.array([], dtype=np.int64)
    val_labels = np.array([], dtype=np.float32)
    val_idx = np.array([], dtype=np.int64)
    test_labels = np.array([], dtype=np.float32)
    test_idx = np.array([], dtype=np.int64)
    unlabeled_labels = np.array([], dtype=np.float32)
    unlabeled_idx = np.array([], dtype=np.int64)
    split_num = num_split_idx // 10
    for _ in range(split_num):
        subdataset = random.choice(dataset_index)
        test_idx = np.concatenate((test_idx, dataset_spilt[subdataset]))
        test_labels = np.concatenate((test_labels, labels_idx[subdataset]))
        test_labels = torch.tensor(test_labels)
        dataset_index.remove(subdataset)
    for _ in range(split_num):
        subdataset = random.choice(dataset_index)
        val_idx = np.concatenate((val_idx, dataset_spilt[subdataset]))
        val_labels = np.concatenate((val_labels, labels_idx[subdataset]))
        val_labels = torch.tensor(val_labels)
        dataset_index.remove(subdataset)

    num = int(train_num * split_num)
    for _ in range(num):
        subdataset = random.choice(dataset_index)
        train_idx = np.concatenate((train_idx, dataset_spilt[subdataset]))
        train_labels = np.concatenate((train_labels, labels_idx[subdataset]))
        train_labels = torch.tensor(train_labels)
        dataset_index.remove(subdataset)

    for i in range(len(dataset_index)):
        unlabeled_idx = np.concatenate((unlabeled_idx, dataset_spilt[dataset_index[i]]))
        unlabeled_labels = np.concatenate((unlabeled_labels, labels_idx[dataset_index[i]]))
        unlabeled_labels = torch.tensor(unlabeled_labels)
    return train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, unlabeled_idx, unlabeled_labels