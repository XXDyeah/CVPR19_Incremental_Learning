#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
# from utils_pytorch import *

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure the feature extractor operates on the correct device to avoid
    # mismatched tensor/weight types when the inputs are on GPU.
    tg_feature_model = tg_feature_model.to(device)
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = np.zeros(input_data.shape[0])
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            # ``tg_feature_model`` returns a tensor that lives on the same device as
            # ``inputs`` (normally GPU).  ``numpy`` however expects CPU based arrays
            # and will implicitly call ``Tensor.__array__`` when given a tensor.  If
            # the tensor is still on the GPU this results in the "can't convert
            # cuda:0 device type tensor to numpy" error.  Move the tensor back to the
            # host before converting it to a ``numpy`` array.
            outputs = tg_feature_model(inputs)
            outputs = outputs.detach().cpu().numpy()
            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(outputs)
            start_idx = start_idx + inputs.shape[0]
    assert(start_idx==num_samples)
    return features
