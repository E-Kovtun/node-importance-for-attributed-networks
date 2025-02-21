export CUDA_VISIBLE_DEVICES=0
mkdir prepared_data
echo "Preparing Cora..."
python src/data_preparation/prepare_cora.py
echo "Preparing CiteSeer..."
python src/data_preparation/prepare_citeseer.py
echo "Preparing PubMed..."
python src/data_preparation/prepare_pubmed.py
echo "Preparing Wiki-CS..."
python src/data_preparation/prepare_wikics.py
echo "Preparing HEP-TH..."
python src/data_preparation/prepare_hepth.py
echo "Preparing DBLP..."
python src/data_preparation/prepare_dblp.py
