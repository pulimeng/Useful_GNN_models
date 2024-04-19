# Some useful GNN Models

This repository contains an integrated module for Graph Neural Networks (GNNs), along with example scripts. All models are implemented with PyG The module is designed to fit data representations and formattings specific to your task. However, some lines in the code will require modifications to ensure a smooth run. Please read the following instructions to understand the purpose of each file and make the necessary changes for successful training. 

## Files Description:

### `data_utils.py`:

This file includes data utility scripts to handle your dataset. The module utilizes the Dataset object from the Torch Geometric library, which is a popular choice for working with graph data in PyTorch. It is essential to modify this file to preprocess and load your specific graph data in the desired format. Follow the comments in the script to understand its functionalities and adapt it to your dataset.

### `models.py`:

Here, you will find implementations of various GNN models, including:
- Graph Convolutional Network (GCN)
- Graph Isomorphism Network (GIN)
- Graph Attention Network (GAT)
- Message Passing Neural Network (MPNN)
- Graph Equivariant Network (GEN)

The file also includes JK-layer (Jumping Knowledge layer) and other global pooling methods, which can be useful components for certain GNN architectures. Review this file to understand the structure and architecture of each model and select the one that best suits your task.

### `train.py`:

This is the main training code responsible for training the GNN model using your dataset and chosen model architecture. During training, this script will output the following metrics:
- Training loss
- Validation loss
- Validation accuracy
- Validation confusion matrix

To train the model, you need to provide specific command-line arguments as follows:

```
python train.py --input_folder .... --labels .... --model model_name --output_folder ./results
```

Replace the placeholders:
- `....` with the actual paths or values for the `input_folder` (where your preprocessed graph data is stored) and `labels` (dataframe contain names of the instances file and labels).
- `model_name` with the name of the GNN model you want to use, such as "GCN," "GIN," "GAT," "MPNN," or "GEN."
- `./results` with the output folder path where the training results will be saved. Feel free to change this path to a different directory if needed.

Before running the training script, ensure you have installed the required dependencies, including PyTorch, Torch Geometric, and any other libraries mentioned in the code.

## Usage:

1. Review the provided files and comments carefully to understand their functionalities and purpose.
2. Modify `data_utils.py` to preprocess and load your specific graph data in the required format.
3. Select an appropriate GNN model from `models.py` that best fits your task and dataset.
4. Adjust the hyperparameters and settings in `params.json` to suit your dataset and training preferences.
5. Run the training script (`train.py`) with the necessary command-line arguments to start the training process.

## Additional Notes:

- Please ensure to follow good coding practices and document any significant changes you make to the code.
- If you encounter any issues or have further questions, don't hesitate to refer to the original documentation or create an issue in the repository for support.

Happy training! If you need further assistance or have any questions, feel free to reach out.
