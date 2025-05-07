# datasets/close_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Optional

class CloSeDataset(Dataset):
    def __init__(self, file_list: List[str]):
        self.files = [Path(f) for f in file_list]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        return {
            'points': torch.tensor(data['points'], dtype=torch.float32),  # (N, 3)
            'scalar_features': torch.tensor(
                np.concatenate([data['colors'], data['normals']], axis=-1),
                dtype=torch.float32),  # (N, 6)
            'pose': torch.tensor(data['pose'], dtype=torch.float32),      # (72,)
            'garments': torch.tensor(data['garments'], dtype=torch.float32),  # (18,)
            'labels': torch.tensor(data['labels'], dtype=torch.long) if 'labels' in data else None
        }

def close_collate(batch):
    result = {}
    for key in batch[0].keys():
        if batch[0][key] is None:
            result[key] = None
            continue

        # inputs: B x ...
        if batch[0][key].ndim == 2:  # (N, C)
            result[key] = torch.stack([x[key] for x in batch], dim=0)
        elif batch[0][key].ndim == 1:  # (C,)
            result[key] = torch.stack([x[key] for x in batch], dim=0)
        else:
            raise ValueError(f"Unexpected ndim for key '{key}': {batch[0][key].ndim}")
    return result
