from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
import torch
import os 

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dblp_dict = {}
    vocab_size = 768

    with open('./data/DBLP/outputacm.txt', 'r') as f:
        content_lines = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)
    model = model.to(device)

    end_indices = []
    for i in range(len(content_lines)):
        if content_lines[i] == '\n':
            end_indices.append(i)

    start = 1
    for end_ind in tqdm(end_indices):
        block = content_lines[start:end_ind]
        for l in block:
            title = block[0]
            assert '#*' in title
            assert len(title) > 0
            title = [title.split('#*')[1].split('\n')[0]]
            if '#index' in l:
                node = l.split('#index')[1].split('\n')[0]
                inputs = tokenizer(title, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)
                inputs = inputs.to(device)
                output = model(**inputs)
                embedding = output.last_hidden_state[:, 0, :]
                embedding = embedding[0, :].detach().cpu().numpy().tolist()
                dblp_dict[node] = {'emb': embedding}
                dblp_dict[node]['out'] = []
            if '#%' in l:
                dblp_dict[node]['out'].append(l.split('#%')[1].split('\n')[0])
        start = end_ind + 1

    year_pos = ['#t' in line for line in content_lines]
    node_years = {content_lines[i+2].split('#index')[1].split('\n')[0]: int(l.split('#t')[1].split('\n')[0])
                for i, l in enumerate(content_lines) if year_pos[i]}

    dblp_dict['520250']['out'] = []

    # Consider papers only with year <= 1990 to find core ones
    dblp_dict_reduced = {}
    for node in dblp_dict.keys():
        if node_years[node] <= 1990:
            dblp_dict_reduced[node] = {'emb': dblp_dict[node]['emb'], 'out': []}

    for cited_node in dblp_dict_reduced.keys():
        for citing_node in dblp_dict[cited_node]['out']:
            if citing_node in dblp_dict_reduced.keys():
                dblp_dict_reduced[citing_node]['out'].append(cited_node)

    with open('./prepared_data/dblp_dict.json', 'w') as f:
        json.dump(dblp_dict_reduced, f)


if __name__ == "__main__":
    main()