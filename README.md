# PINE🌲: Pipeline for Important Node Exploration in Attributed Networks

PINE is an unsupervised approach for identifying important nodes in attributed networks. Importance of nodes is considered within an influence maximization paradigm. The nodes are vital if they cause a great information spread in the network if a diffusion process starts from them. PINE allows to effectively account for node features to discover crucial nodes from topology and attribute perspectives. PINE is a learning-based framework with an underlying attention mechanism, which is a key difference from the traditional centrlaity measures, e.g. Degree Centrality or PageRank. 

![pine](./pictures/pine.png)

To find important nodes in the attributed graphs, proceed as follows.

### Step 1. Set environment
To start, create conda environment with the name 'pine_env' with proper dependencies by running: 
```
conda env create --file=environment.yml
```
Then, activate it:
```
conda activate pine_env
```


### Step 2. Get data

We provide the studied datasets in https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link. There are six attributed datasets: Cora, CiteSeer, PubMed, Wiki-CS, HEP-TH, and DBLP.
To download all of them into folder `data`, run:
```
bash bin/get_data.sh
```

### Step 3. Prepare data

We convert all datasets to the dict format with an unified structure. Note that for some datasets, we get embeddings from text attributes with the help of the pretrained models from HuggingFace.  
To prepare datasets and save them into folder `prepare_data`, run:
```
bash bin/prepare_data.sh
```

The resulting dicts have node id as a key and information on that node as a value. Information on each node include:
* `emb` - an embedding vector of a text attribute of a node
* `out` - list of nodes for which the considered node is an information supplier. It means that there is a knowledge flow from the condifered node to all nodes from the list. 



