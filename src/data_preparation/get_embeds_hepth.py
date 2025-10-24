from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import os 
import torch

def main():
    physics_dict = {}
    vocab_size = 768 

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("thellert/physbert_cased")
    model = AutoModel.from_pretrained("thellert/physbert_cased")
    model = model.to(device)

    meta_folder = './data/HEP-TH/cit-HepTh-abstracts/'
    year_folder_names = os.listdir(meta_folder)
    year_folder_names = [n for n in year_folder_names if not n.startswith('.')]
    for year_folder in year_folder_names:
        year_files = os.listdir(os.path.join(meta_folder, year_folder))
        for filename in tqdm(year_files):
            node = filename.split('.')[0]
            with open(os.path.join(meta_folder, year_folder, filename), 'r') as f:
                content_lines = f.readlines()
            title_line = [line for line in content_lines if 'Title' in line] ### Abstract can be used instead of title
            title = title_line[0].split('Title: ')[1].split('\n')[0]
            inputs = tokenizer(title, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            token_embeddings = token_embeddings[:, 1:-1, :]
            sentence_embedding = token_embeddings.mean(dim=1)
            physics_dict[node] = {'emb': sentence_embedding[0, :].detach().cpu().numpy().tolist()}
            
    with open('./data/HEP-TH/Cit-HepTh.txt', 'r') as f:
        cite_lines = f.readlines()

    for n in physics_dict:
        physics_dict[n]['out'] = []

    for i in tqdm(range(4, len(cite_lines))):
        ind_split0 = cite_lines[i].find('\t')
        ind_split1 = cite_lines[i].find('\n')
        paper1 = cite_lines[i][:ind_split0]
        paper2 = cite_lines[i][ind_split0+1:ind_split1]
        # paper1 --> paper2
        if (paper1 in physics_dict) and (paper2 in physics_dict):
            physics_dict[paper2]['out'].append(paper1) 

    with open('./data/hepth_dict.json', 'w') as f:
        json.dump(physics_dict, f)


if __name__ == "__main__":
    main()