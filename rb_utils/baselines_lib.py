# This library contains various gradient estimators
# including REINFORCE, REINFORCE+,
# REBAR/RELAX, NVIL, and gumbel_softmax

import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Categorical
from common_utils import get_one_hot_encoding_from_int, sample_class_weights

import torch.nn.functional as F

import gumbel_softmax_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_reinforce_grad_sample(conditional_loss, class_weights,
                                baseline = 0.0):
    # computes the REINFORCE gradient estimate
    assert len(conditional_loss) == len(class_weights)

    nz = (class_weights > 0).to(class_weights.device)

    out = torch.where(
                nz,
                (conditional_loss - baseline).detach() * torch.log(class_weights),
                torch.tensor(0., device=class_weights.device, dtype=torch.float))

    return out


"""
Below are the gradient estimates for
REINFORCE, REINFORCE+, REBAR/RELAX, NVIL, and Gumbel-softmax.
Each follow the pattern,

Parameters
----------
conditional_loss_fun : function
    A function that returns the loss conditional on an instance of the
    categorical random variable. It must take in a one-hot-encoding
    matrix (batchsize x n_categories) and return a vector of
    losses, one for each observation in the batch.
log_class_weights : torch.Tensor
    A tensor of shape batchsize x n_categories of the log class weights
class_weights_detached : torch.Tensor
    A tensor of shape batchsize x n_categories of the class weights.
    Must be detached, i.e. we do not compute gradients
seq_tensor : torch.Tensor
    A tensor containing values \{1 ... batchsize\}
    TODO: is there a way to cache this?
z_sample : torch.Tensor
    The cateories (not one-hot-encoded) at which to evaluate the ps loss.
epoch : int
    The epoch of the optimizer (for Gumbel-softmax, which has an annealing rate)
data : torch.Tensor
    The data at which we evaluate the loss (for NVIl and RELAX, which have
    a data dependent baseline)
grad_estimator_kwargs : dict
    Additional arguments to the gradient estimators

Returns
-------
ps_loss :
    a value such that ps_loss.backward() returns an
    estimate of the gradient.
    In general, ps_loss might not equal the actual loss.
"""

def reinforce(conditional_loss_fun, class_weights,
                class_weights_detached, seq_tensor,
                z_sample, epoch, data, grad_estimator_kwargs = None):
    # z_sample should be a vector of categories
    # conditional_loss_fun is a function that takes in a one hot encoding
    # of z and returns the loss

    assert len(z_sample) == class_weights.shape[0]

    # compute loss from those categories
    n_classes = class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == class_weights.shape[0]

    # get log class_weights
    class_weights_i = class_weights[seq_tensor, z_sample]

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    class_weights_i, baseline = 0.0) + \
                        conditional_loss_fun_i

def reinforce_w_double_sample_baseline(\
            conditional_loss_fun, class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            grad_estimator_kwargs = None):
    # This is what we call REINFORCE+ in our paper,
    # where we use a second, independent sample from the discrete distribution
    # to use as a baseline

    assert len(z_sample) == class_weights.shape[0]

    # compute loss from those categories
    n_classes = class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == class_weights.shape[0]

    # get log class_weights
    class_weights_i = class_weights[seq_tensor, z_sample]

    # get baseline
    z_sample2 = sample_class_weights(class_weights_detached)
    one_hot_z_sample2 = get_one_hot_encoding_from_int(z_sample2, n_classes)
    baseline = conditional_loss_fun(one_hot_z_sample2)

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    class_weights_i, baseline) + conditional_loss_fun_i

class RELAXBaseline(nn.Module):
    def __init__(self, input_dim):
        # this is a neural network for the NVIL baseline
        super(RELAXBaseline, self).__init__()

        # image / model parameters
        self.input_dim = input_dim

        # define the linear layers
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        return h

def relax(conditional_loss_fun, class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            temperature = torch.Tensor([1.0]),
            eta = 1.,
            c_phi = lambda x : torch.Tensor([0.0])):
    # with the default c_phi value, this is just REBAR
    # RELAX adds a learned component c_phi

    log_class_weights = torch.log(class_weights)
    # sample gumbel
    gumbel_sample = log_class_weights + \
        gumbel_softmax_lib.sample_gumbel(log_class_weights.size())

    # get hard z
    _, z_sample = gumbel_sample.max(dim=-1)
    n_classes = log_class_weights.shape[1]
    z_one_hot = get_one_hot_encoding_from_int(z_sample, n_classes)
    temperature = torch.clamp(temperature, 0.01, 5.0)

    # get softmax z
    z_softmax = F.softmax(gumbel_sample / temperature[0], dim=-1)

    # conditional softmax z
    z_cond_softmax = \
        gumbel_softmax_lib.gumbel_softmax_conditional_sample(\
            log_class_weights, temperature[0], z_one_hot)

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    # reinforce term
    f_z_hard = conditional_loss_fun(z_one_hot.detach())
    f_z_softmax = conditional_loss_fun(z_softmax)
    f_z_cond_softmax = conditional_loss_fun(z_cond_softmax)

    # baseline terms
    c_softmax = c_phi(z_softmax).squeeze()
    z_cond_softmax_detached = \
        gumbel_softmax_lib.gumbel_softmax_conditional_sample(\
            log_class_weights, temperature[0], z_one_hot, detach = True)
    c_cond_softmax = c_phi(z_cond_softmax_detached).squeeze()

    reinforce_term = \
        (f_z_hard - eta * (f_z_cond_softmax - c_cond_softmax)).detach() * \
                        log_class_weights_i + \
                        log_class_weights_i * eta * c_cond_softmax

    # correction term
    correction_term = eta * (f_z_softmax - c_softmax) - \
                        eta * (f_z_cond_softmax - c_cond_softmax)

    return reinforce_term + correction_term + f_z_hard

def gumbel(conditional_loss_fun, class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            annealing_fun,
            straight_through = True):

    log_class_weights = torch.log(class_weights)
    # get temperature
    temperature = annealing_fun(epoch)

    # sample gumbel
    if straight_through:
        gumbel_sample = \
            gumbel_softmax_lib.gumbel_softmax(log_class_weights, temperature)
    else:
        gumbel_sample = \
            gumbel_softmax_lib.gumbel_softmax_sample(\
                    log_class_weights, temperature)

    f_gumbel = conditional_loss_fun(gumbel_sample)

    return f_gumbel

class BaselineNN(nn.Module):
    def __init__(self, slen = 28):
        # this is a neural network for the NVIL baseline
        super(BaselineNN, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.slen = slen

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)


    def forward(self, image):

        # feed through neural network
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return h


def nvil(conditional_loss_fun, class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            baseline_nn):

    assert len(z_sample) == class_weights.shape[0]

    # compute loss from those categories
    n_classes = class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == class_weights.shape[0]

    # get log class_weights
    class_weights_i = class_weights[seq_tensor, z_sample]

    # get baseline
    baseline = baseline_nn(data).squeeze()

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    class_weights_i, baseline = baseline) + \
                        conditional_loss_fun_i + \
                        (conditional_loss_fun_i.detach() - baseline)**2
