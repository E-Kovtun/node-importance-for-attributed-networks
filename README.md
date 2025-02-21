# PINE🌲: Pipeline for Important Node Exploration in Attributed Networks

PINE is an unsupervised approach for identifying important nodes in attributed networks. Importance of nodes is considered within an influence maximization paradigm. The nodes are vital if they cause a great information spread in the network if a diffusion process starts from them. PINE allows to effectively account for node features to discover crucial nodes from topology and attribute perspectives. PINE is a learning-based framework with an underlying attention mechanism, which is a key difference from the traditional centrlaity measures, like Degree Centrality or PageRank. 

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
Also, add the root path of this repo to PYTHONPATH.


### Step 2. Get data

We provide the studied datasets in https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link. There are six attributed datasets: Cora, CiteSeer, PubMed, Wiki-CS, HEP-TH, and DBLP.
To download all of them into folder `data`, run:
```
bash bin/get_data.sh
```

### Step 3. Prepare data

We convert all datasets to the dict format with an unified structure. Note that for some datasets, we get embeddings from text attributes with the help of the pretrained models from HuggingFace.  
To prepare datasets and save them into folder `prepared_data`, run:
```
bash bin/prepare_data.sh
```

The resulting dicts have node id as a key and information on that node as a value. Information on each node include:
* `emb` - an embedding vector of a text attribute of a node
* `out` - list of nodes for which the considered node is an information supplier. It means that there is a knowledge flow from the considered node to each node from the list.

### Step 4. Run methods

A folder `src/baselines` includes different methods for ealuation of node importance. A folder `src/pine` contains an implementation of our approach PINE, entaling a train process of attention-bassed graph network on Link Prediction task. Noteworthy, there are no need in any external markup, as Link Prediction is an unsupervised task. 

#### Output of methods
A goal of each method is to assign each node with an importance score calculted with the predefined rule. The larger assigned score the more important the node is considered to be in the network. The limitation of baseline approaches is their exclusive focus on structural properties. However, to estimate an importance of each node in the attributed network comprehensively, we need to account for information in node features. PINE is capable of doing that. 

#### Evaluation of results
To compare method performance, we adopt simulation-based procedure. Each methods associates importnace scores with nodes. Then, top-K nodes are taken as seed node for the start of information diffusion process. At the end, an influence spread over the network is evaluated. The method is better than others if it identifies nodes, which lead to the greatest spread of information. 

#### Simulation models
As we consider attributed networks, it is important to simulate information propagation taking into account node attributes. To do this, each edge of the graph is associated with topology and attribute weghts. Then, these weights are used in such simulaiton models as **Linear Threshold (LT)** and **Independent Cascade (IC)**. We mark them as **LT+** and **IC+** to indicate their attribute-awareness. Their implementations are given in a folder `src/simulation`.

#### Launch all methods
To run methods:
```
python src/run_methods.py
```

In `src/run_methods.py`, you need to specify a path to a dataset, a number of seed nodes (`num_starts`) for initiation of diffusion process, and a number of simulation runs `num_runs` (there will be `num_runs` simulation runs for each value of `num_starts`). 

As an output, you will get a dict in `results` folder, with the keys `lt` and `ic`, denoting a propagation model. For each propagation model, there are names of the considered methods and the corresponding values of influence spread. A number of influence spread values for each method is defined by a number of values in `num_starts`. 
