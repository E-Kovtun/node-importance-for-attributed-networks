from tqdm import tqdm
import re
import json

def main():
    with open('./data/CiteSeer/citeseer.content', 'r') as f:
        content_lines = f.readlines()

    vocab_size = 3703
    citeseer_dict = {}

    for i in tqdm(range(len(content_lines))):
        ind_split0 = content_lines[i].find('\t')
        ind_split1 = content_lines[i].rfind('\t')
        emb = list(re.sub('\t', '', content_lines[i][ind_split0+1:ind_split1]))
        assert len(emb) == vocab_size
        citeseer_dict[content_lines[i][:ind_split0]] = {'emb': emb, 'label': re.sub('\n', '', content_lines[i][ind_split1+1:])}

    with open('./data/CiteSeer/citeseer.cites', 'r') as f:
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

    with open('./prepared_data/citeseer_dict.json', 'w') as f:
        json.dump(citeseer_dict, f)


if __name__ == "__main__":
    main()


