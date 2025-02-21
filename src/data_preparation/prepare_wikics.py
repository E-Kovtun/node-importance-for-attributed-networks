from tqdm import tqdm
import json

def main():
    with open('./data/Wiki-CS/data.json', 'r') as f:
        data = json.load(f)

    with open('./data/Wiki-CS/metadata.json', 'r') as f:
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

    with open('./prepared_data/wikics_dict.json', 'w') as f:
        json.dump(wikipedia_dict, f)


if __name__ == "__main__":
    main()
