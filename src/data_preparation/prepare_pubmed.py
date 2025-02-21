from tqdm import tqdm
import re
import json

def main():
    with open('./data/PubMed/Pubmed.content', 'r') as f:
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

    with open('./data/PubMed/Pubmed.cites', 'r') as f:
        cite_lines = f.readlines()

    for n in pubmed_dict:
        pubmed_dict[n]['out'] = []

    for line_num in range(2, len(cite_lines)):
        paper1 = cite_lines[line_num].split(':')[1].split('\t')[0]
        paper2 = cite_lines[line_num].split(':')[2].split('\n')[0]
        pubmed_dict[paper1]['out'].append(paper2) 

    with open('./prepared_data/pubmed_dict.json', 'w') as f:
        json.dump(pubmed_dict, f)


if __name__ == "__main__":
    main()
