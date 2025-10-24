import pandas as pd
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import os
from tqdm import tqdm
import re
import json
import dgl
from torch_geometric.data import Data
import torch


def get_graph(dataset_name):

    if dataset_name == 'ogbn-arxiv':
        dataset_path = None
        feature_path = None

        data = NodePropPredDataset(name='ogbn-arxiv', root='./data/')
        graph, labels = data[0]
        labels = labels[:, 0] # for solving node classification task
        graph_edges = graph['edge_index']
        feat = graph['node_feat']
        return feat, graph_edges
    
    elif dataset_name == 'cora':
        dataset_path = './data/Cora'
        feature_path = None
        with open(os.path.join(dataset_path, 'cora.content'), 'r') as f:
            content_lines = f.readlines()

        vocab_size = 1433
        cora_dict = {}

        for i in tqdm(range(len(content_lines))):
            ind_split0 = content_lines[i].find('\t')
            ind_split1 = content_lines[i].rfind('\t')
            emb = list(re.sub('\t', '', content_lines[i][ind_split0+1:ind_split1]))
            assert len(emb) == vocab_size
            cora_dict[content_lines[i][:ind_split0]] = {'emb': emb, 'label': re.sub('\n', '', content_lines[i][ind_split1+1:])}

        with open(os.path.join(dataset_path, 'cora.cites'), 'r') as f:
            cite_lines = f.readlines()

        for n in cora_dict:
            cora_dict[n]['out'] = []

        for i in tqdm(range(len(cite_lines))):
            ind_split0 = cite_lines[i].find('\t')
            ind_split1 = cite_lines[i].find('\n')
            paper1 = cite_lines[i][:ind_split0]
            paper2 = cite_lines[i][ind_split0+1:ind_split1]
            # paper2 --> paper1
            cora_dict[paper1]['out'].append(paper2) 

        for u in cora_dict:
            cora_dict[u]['emb'] = list(map(int, cora_dict[u]['emb']))

        name2id = {}
        id = 0
        feat = []
        graph_edges = []
        for u in cora_dict:
            for v in cora_dict[u]['out']:
                if u not in name2id:
                    name2id[u] = id
                    id += 1
                if v not in name2id:
                    name2id[v] = id
                    id += 1
                graph_edges.append([name2id[v], name2id[u]])
        id2name = {v: k for k, v in name2id.items()}
        for id in range(len(id2name)):
            feat.append(cora_dict[id2name[id]]['emb'])

        feat = np.array(feat).astype(np.float32)
        graph_edges = np.array(graph_edges).transpose().astype(np.int64)

        return feat, graph_edges
        
    elif dataset_name == 'citeseer':
        dataset_path = './data/CiteSeer'
        feature_path = None
        with open(os.path.join(dataset_path, 'citeseer.content'), 'r') as f:
            content_lines = f.readlines()
        vocab_size = 3703
        citeseer_dict = {}

        for i in tqdm(range(len(content_lines))):
            ind_split0 = content_lines[i].find('\t')
            ind_split1 = content_lines[i].rfind('\t')
            emb = list(re.sub('\t', '', content_lines[i][ind_split0+1:ind_split1]))
            assert len(emb) == vocab_size
            citeseer_dict[content_lines[i][:ind_split0]] = {'emb': emb, 'label': re.sub('\n', '', content_lines[i][ind_split1+1:])}

        with open(os.path.join(dataset_path, 'citeseer.cites'), 'r') as f:
            cite_lines = f.readlines()

        for n in citeseer_dict:
            citeseer_dict[n]['out'] = []

        for i in tqdm(range(len(cite_lines))):
            ind_split0 = cite_lines[i].find('\t')
            ind_split1 = cite_lines[i].find('\n')
            paper1 = cite_lines[i][:ind_split0]
            paper2 = cite_lines[i][ind_split0+1:ind_split1]
            # paper2 --> paper1
            if (paper1 in citeseer_dict) and (paper2 in citeseer_dict):
                citeseer_dict[paper1]['out'].append(paper2) 
            else:
                continue

        for u in citeseer_dict:
            citeseer_dict[u]['emb'] = list(map(int, citeseer_dict[u]['emb']))

        name2id = {}
        id = 0
        feat = []
        graph_edges = []
        for u in citeseer_dict:
            for v in citeseer_dict[u]['out']:
                if u not in name2id:
                    name2id[u] = id
                    id += 1
                if v not in name2id:
                    name2id[v] = id
                    id += 1
                graph_edges.append([name2id[v], name2id[u]])
        id2name = {v: k for k, v in name2id.items()}
        for id in range(len(id2name)):
            feat.append(citeseer_dict[id2name[id]]['emb'])

        feat = np.array(feat).astype(np.float32)
        graph_edges = np.array(graph_edges).transpose().astype(np.int64)

        return feat, graph_edges
    
    elif dataset_name == 'pubmed':
        dataset_path = './data/PubMed'
        feature_path = None
        with open(os.path.join(dataset_path, 'Pubmed.content'), 'r') as f:
            content_lines = f.readlines()

        init_features = content_lines[1].split('\t')[1:-1]
        feature_order = [feat.split('numeric:')[1].split(':0.0')[0] for feat in init_features]

        vocab_size = 500
        pubmed_dict = {}

        for line_num in range(2, len(content_lines)):
            content = content_lines[line_num].split('\t')
            node = content[0]
            label = content[1].split('=')[1]
            features = [fl.split('=') for fl in content[2:-1]]
            feature_dict = {feat: float(value) for [feat, value] in features}
            pubmed_dict[node] = {'emb': [feature_dict[feat] if feat in feature_dict else 0 for feat in feature_order], 'label': label}

        with open(os.path.join(dataset_path, 'Pubmed.cites'), 'r') as f:
            cite_lines = f.readlines()

        for n in pubmed_dict:
            pubmed_dict[n]['out'] = []

        for line_num in range(2, len(cite_lines)):
            paper1 = cite_lines[line_num].split(':')[1].split('\t')[0]
            paper2 = cite_lines[line_num].split(':')[2].split('\n')[0]
            pubmed_dict[paper1]['out'].append(paper2) 

        name2id = {}
        id = 0
        feat = []
        graph_edges = []
        for u in pubmed_dict:
            for v in pubmed_dict[u]['out']:
                if u not in name2id:
                    name2id[u] = id
                    id += 1
                if v not in name2id:
                    name2id[v] = id
                    id += 1
                graph_edges.append([name2id[v], name2id[u]])
        id2name = {v: k for k, v in name2id.items()}
        for id in range(len(id2name)):
            feat.append(pubmed_dict[id2name[id]]['emb'])

        feat = np.array(feat).astype(np.float32)
        graph_edges = np.array(graph_edges).transpose().astype(np.int64)

        return feat, graph_edges
    
    elif dataset_name == 'wiki-cs':
        dataset_path = './data/Wiki-CS'
        feature_path = None
        with open(os.path.join(dataset_path, 'data.json'), 'r') as f:
            data = json.load(f)

        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        wikipedia_dict = {}
        vocab_size = 300

        for i in range(len(metadata['nodes'])):
            node = metadata['nodes'][i]['id']
            wikipedia_dict[node] = {'emb': data['features'][i]}

        for node in wikipedia_dict.keys():
            wikipedia_dict[node]['out'] = []

        for i in range(len(metadata['nodes'])):
            citing_node = metadata['nodes'][i]['id']
            for cited_node in metadata['nodes'][i]['outlinks']:
                wikipedia_dict[cited_node]['out'].append(citing_node)

        name2id = {}
        id = 0
        feat = []
        graph_edges = []
        for u in wikipedia_dict:
            for v in wikipedia_dict[u]['out']:
                if u not in name2id:
                    name2id[u] = id
                    id += 1
                if v not in name2id:
                    name2id[v] = id
                    id += 1
                graph_edges.append([name2id[v], name2id[u]])
        id2name = {v: k for k, v in name2id.items()}
        for id in range(len(id2name)):
            feat.append(wikipedia_dict[id2name[id]]['emb'])

        feat = np.array(feat).astype(np.float32)
        graph_edges = np.array(graph_edges).transpose().astype(np.int64)

        return feat, graph_edges
    
    elif dataset_name == 'dblp':
        dataset_path = './data/DBLP/Citation-2015.pt'
        feature_path = './data/DBLP/Citation_roberta_base_512_cls.npy'
        g = dgl.load_graphs(dataset_path)[0][0]

        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)
    
        graph_edges = data.edge_index.numpy()
        feat = np.load(feature_path).astype(np.float32)
        return feat, graph_edges
    
    elif dataset_name == 'hepth':
        dataset_path = './data/HEP-TH/hepth_dict.json'
        feature_path = None
        with open(dataset_path, 'r') as f:
            hepth_dict = json.load(f)

        name2id = {}
        id = 0
        feat = []
        graph_edges = []
        for u in hepth_dict:
            for v in hepth_dict[u]['out']:
                if u not in name2id:
                    name2id[u] = id
                    id += 1
                if v not in name2id:
                    name2id[v] = id
                    id += 1
                graph_edges.append([name2id[v], name2id[u]])
        id2name = {v: k for k, v in name2id.items()}
        for id in range(len(id2name)):
            feat.append(hepth_dict[id2name[id]]['emb'])

        feat = np.array(feat).astype(np.float32)
        graph_edges = np.array(graph_edges).transpose().astype(np.int64)

        return feat, graph_edges
    
    else:
        print('Unknown dataset')

