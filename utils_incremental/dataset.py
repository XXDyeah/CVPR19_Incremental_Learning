import torch
from torch.utils.data import Dataset

class DatasetWithIndex(Dataset):
    """Dataset wrapper that also returns the sample index.
    This is needed for TIAW weighting to keep per-sample prediction
    history. It behaves like the wrapped dataset but each ``__getitem__``
    returns ``(image, target, index)``.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return image, target, idx
