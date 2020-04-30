import numpy as np
import torch
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot_encoding_from_int(z, n_classes):
    """
    Convert categorical variable to one-hot enoding

    Parameters
    ----------
    z : torch.LongTensor
        Tensor with integers corresponding to categories
    n_classes : Int
        The total number of categories

    Returns
    ----------
    z_one_hot : torch.Tensor
        One hot encoding of z
    """

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

def sample_class_weights(class_weights):
    """
    draw a sample from Categorical variable with
    probabilities class_weights
    """

    nz_mask = class_weights.sum(1) != 0
    samples = torch.zeros(nz_mask.shape, dtype=torch.long)
    for i, elem in enumerate(nz_mask):
        if elem:
            samples[i] = np.random.choice(
                torch.arange(class_weights.shape[1]),
                p=class_weights[i].cpu().numpy())

    return samples.to(class_weights.device).detach()
