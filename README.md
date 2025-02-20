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

We provide the studied datasets in https://drive.google.com/drive/folders/1ZYhW_FixUBKTDjegfbZKg62VrEAkt0gR?usp=sharing.
To download all datasets into folder `data`, run:
```
bash bin/get_data.sh
```
