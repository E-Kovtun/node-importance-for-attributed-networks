# PINE🌲: Pipeline for Important Node Exploration in Attributed Networks

PINE is an unsupervised approach for identifying important nodes in attributed networks. Importance of nodes is considered within an Influence Maximization (IM) paradigm. In the context of IM, the nodes are vital if they cause a great information spread in the network when a knowledge dissemination starts from them. PINE allows to effectively account for node features to discover crucial nodes from topology and attribute perspectives. PINE framework includes training of Graph Attention Network (GAT) to solve Link Prediction task for a subsequent use of attention distribution in node importance estimation. A presence of learning component is a key difference from the traditional centrlaity measures, like Degree Centrality or PageRank. 

![pine](./pictures/pine_scheme1.png)

In summary, the result of PINE work is an identified set of important nodes in view of graph structure and node attributes:

![graph](./pictures/graph_citeseer1.png)

# 🚀 Launch PINE


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

We provide the studied datasets in [https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link](https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link). Seven attributed homogeneous networks are under consideration: Cora, CiteSeer, PubMed, Wiki-CS, HEP-TH, ogbn-Arxiv, and DBLP.
To download them into folder `data`, run:
```
bash bin/get_data.sh
```
ogbn-Arxiv dataset is available from [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/). 

The datasets, like Cora, CiteSeer, PubMed, Wiki-CS, and ogbn-Arxiv come with already prepared embeddings for text attributes in nodes. For the DBLP dataset, we use graph with node embeddings prepared in [TAG-benchmark](https://github.com/sktsherlock/TAG-Benchmark) (roberta_base_512_cls model). For HEP-TH dataset, we utilize [PhysBERT](https://huggingface.co/thellert/physbert_cased) model to infer embeddings (check [python script](src/data_preparation/get_embeds_hepth.py) for that, transformers library is needed). 

### Step 3. Run methods
Here, we compare PINE with a set of traditional centrality measures. 

**Problem setup**. A goal of each method is to associate every node in a graph with an importance score. The larger assigned score the more important the node is considered to be in the network. 

**Evaluation of results**. 
To compare methods' performance, we adopt a simulation-based procedure. Each method associates importance scores with nodes. Then, top-K nodes are taken as seed nodes for the start of information diffusion process. At the end, an influence spread over the network is evaluated. We assume that the method is better than others if it identifies nodes, which lead to the greatest spread of information. 

**Propagation models**.
As we consider attributed networks, it is important to simulate information diffusion taking into account node attributes. To do this, each edge of the graph is associated with topology and attribute weights. Then, these weights are used in such propagation models as **Linear Threshold (LT+)** and **Independent Cascade (IC+)**. Plus sign in their names indicates their attribute-awareness. In addition, we utilize a classical **SIR** propagation model, but it relies only on the graph structure. The implementations of propagation models are given in a folder `src/simulation`: [**LT+**](src/simulation/LT_plus.py), [**IC+**](src/simulation/IC_plus.py), and [**SIR**](src/simulation/SIR.py).

**Launch script**

```
python src/run_methods.py \
--dataset_names 'cora' 'citeseer' 'pubmed' 'wiki-cs' 'hepth' 'ogbn-arxiv' 'dblp' \
--measure_names 'pine' 'degree' 'out-degree' 'weighted' 'relative' 'pagerank' 'voterank' 'katz' 'closeness' 'betweenness' 'entropy_dir' \
--propagation_model_names 'LT+' 'IC+' 'SIR' \
--res_folder './simulation_results' \
--device 'cuda:0' \
--node_ratio 0.1 \
--num_runs 1000
```

* `dataset_names` is names of networks, on which different measures are compared.
* `measure_names` is names of measures for node importance estimation.
* `propagation_model_names` pointa out which propagation models to use for a simulation of information dissemination in a network.
* `device` is used for PINE training. Other graph measures are calculated on cpu by default.
* `node_ratio` is a part of the nodes from which information dissemination process starts. 0.1 means that 10\% of nodes with the greatest importance scores are initialized as active.
* `num_runs` is a number of Monte-Carlo simulation runs.
* `res_folder` will contain the results that include optimized hyperparametrs of GAT, which is trained within PINE, and csv files with influence spread values for the selected measures under the specified propagation models. 





