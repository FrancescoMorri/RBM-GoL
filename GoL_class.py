import torch
from torch.utils.data import Dataset


class GoL(Dataset):

    def __init__(self, grid, state):
        self.inputs = grid
        self.labels = state

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        outputs = torch.tensor(self.inputs[idx], dtype = torch.float)
        labels = torch.tensor(int(self.labels[idx]), dtype = torch.long)
        return outputs, labels