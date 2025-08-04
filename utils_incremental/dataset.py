import torch
from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    """Wrap a dataset to also return the sample index.

    Each ``__getitem__`` returns ``(image, target, index, flag)`` where
    ``flag`` is ``0`` indicating the sample is a real one (from ``D_t`` or
    ``M_{t-1}``).  The ``target`` is an integer class label.
    """

    def __init__(self, dataset, offset: int = 0):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return image, torch.tensor(target, dtype=torch.long), self.offset + idx, torch.zeros(1, dtype=torch.long)


class CounterfactualDataset(Dataset):
    """Dataset that stores pre-synthesised counterfactual samples.

    ``target`` is a soft label vector of shape ``(num_classes,)`` and the
    returned ``flag`` is ``1`` to indicate that the sample is synthetic.
    """

    def __init__(self, images: torch.Tensor, soft_labels: torch.Tensor, offset: int):
        assert images.size(0) == soft_labels.size(0)
        self.images = images
        self.soft_labels = soft_labels
        self.offset = offset

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        # Detach the returned tensors to ensure they do not carry gradient
        # history.  Synthetic samples may be produced by neural networks and thus
        # inadvertently retain ``requires_grad=True``.  When such tensors are
        # collated by ``DataLoader`` (which internally uses ``torch.stack`` with
        # an ``out`` parameter), PyTorch raises an error because automatic
        # differentiation is not supported for that operation.  Returning detached
        # tensors keeps the dataset agnostic to the provenance of the data and
        # prevents unexpected autograd interactions during loading.
        return (self.images[idx].detach(), self.soft_labels[idx].detach(),
                self.offset + idx, torch.ones(1, dtype=torch.long))


def collate_with_soft_targets(batch):
    images, targets, indices, flags = zip(*batch)
    images = torch.stack(images)

    num_classes = None
    for t in targets:
        if t.dim() > 0:
            num_classes = t.size(0)
            break
    if num_classes is None:
        num_classes = int(max(int(t) if t.dim() == 0 else t.argmax().item() for t in targets)) + 1

    processed = []
    for t in targets:
        if t.dim() == 0:
            one_hot = torch.zeros(num_classes, dtype=torch.float)
            one_hot[t.long()] = 1.0
            processed.append(one_hot)
        else:
            processed.append(t.float())
    targets = torch.stack(processed)

    indices = torch.tensor(indices, dtype=torch.long)
    flags = torch.stack(flags)
    return images, targets, indices, flags
