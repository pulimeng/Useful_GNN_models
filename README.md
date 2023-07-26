# CSBG@LSU GNN Models

This repo contains an integrated module for GNNs. All scripts are examples, and it only fits my data representations and formattings. So anyone using this should change accordingly in some lines. I've marked some lines that are most likely to be changed. However, to ensure a smooth run, please read the codes and comments to make necessary changes.

`data_utils.py` is the data utility scripts. It is the Dataset object from Torch Geometric.

`models.py` contains models GCN, GIN, GAT, MPNN, and GEN. JK-layer and other global poolings are also included.

`train.py` is the training code. It outputs train loss, validation loss, validation acc, and validation confusion matrix.

`params.json` is the parameter file for the models and other codes.

To train the model,

`python train.py --input_folder .... --labels .... --model model_name --output_folder ./results`
