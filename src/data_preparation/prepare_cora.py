from tqdm import tqdm
import re
import json

def main():
    with open('./data/Cora/cora.content', 'r') as f:
        content_lines = f.readlines()

    vocab_size = 1433
    cora_dict = {}

    for i in tqdm(range(len(content_lines))):
        ind_split0 = content_lines[i].find('\t')
        ind_split1 = content_lines[i].rfind('\t')
        emb = list(re.sub('\t', '', content_lines[i][ind_split0+1:ind_split1]))
        assert len(emb) == vocab_size
        cora_dict[content_lines[i][:ind_split0]] = {'emb': emb, 'label': re.sub('\n', '', content_lines[i][ind_split1+1:])}

    with open('./data/Cora/cora.cites', 'r') as f:
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

    with open('./prepared_data/cora_dict.json', 'w') as f:
        json.dump(cora_dict, f)


if __name__ == "__main__":
    main()
