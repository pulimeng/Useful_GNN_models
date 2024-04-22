import os.path as osp
import h5py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data.sampler import Sampler

def read_h5_file(path):
    with h5py.File(path, 'r') as f:
        X = f['X'][()]
        eI = f['eI'][()]
        # eAttr = f['eAttr'][()]
        y = f['y'][()]
    X = torch.tensor(X, dtype=torch.float32)
    eI = torch.tensor(eI, dtype=torch.long)
    # eAttr = torch.tensor(eAttr, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=X, edge_index=eI, y=y)
    return data

class MyGraphDataset(Dataset):
    def __init__(self, root, df, device):
        super(MyGraphDataset, self).__init__()
        self.root = root
        self.df = df
        self.device = device
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'instance']  # Assuming 'instance' is the column containing filenames
        data = read_h5_file(osp.join(self.root, filename, filename + '.h5'))  # Corrected filename concatenation
        # Convert data to device if needed
        data = data.to(self.device)
        return data
    
    @staticmethod
    def my_collate_fn(data_list):
        # Collate function to handle batching of Data objects
        batch = Batch.from_data_list(data_list)
        return batch

class CustomImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, logger, label_weights, device='cpu'):
        self.device = torch.device(device)
        self.logger = logger
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)

        self.weights = torch.DoubleTensor(
                    [label_weights[dataset[idx].y.squeeze().detach().item()] for idx in self.indices]
                ).to(self.device)  # Create tensor on CPU and then move to device        
        self.logger.info("Weights calculated and sampler initialized.")

    def __iter__(self):
        # Create your sampling logic here based on self.label_to_count
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
