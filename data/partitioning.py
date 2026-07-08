import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def dirichlet_partition(dataset, num_clients, alpha_dir, num_classes=100, seed=42):
    """
    Partitions a dataset among clients using a Dirichlet distribution to simulate
    severe non-IID data heterogeneity.
    """
    np.random.seed(seed)
    train_labels = np.array(dataset.targets)
    client_indices = {i: [] for i in range(num_clients)}
    
    # Allocate indices for each class based on Dirichlet distribution
    for k in range(num_classes):
        idx_k = np.where(train_labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha_dir, num_clients))
        
        # Distribute indices based on proportions
        proportions = np.array([p * (len(idx_j) < (len(dataset) / num_clients)) for p, idx_j in zip(proportions, client_indices.values())])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_batch = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i] += idx_batch[i].tolist()
            
    return client_indices

def get_client_loaders(dataset, client_indices, batch_size=32):
    """Returns a dictionary of PyTorch DataLoaders for each client."""
    return {
        i: DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
        for i, indices in client_indices.items()
    }