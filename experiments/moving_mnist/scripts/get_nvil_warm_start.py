import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_nvil_baseline_nn(baseline_nn, get_log_q, get_conditional_loss,
                loader, optimizer):

    num_images = len(loader.dataset)

    for batch_idx, data in enumerate(loader):

        optimizer.zero_grad()

        image = data['image'].to(device)

        # get class weights
        log_q = get_log_q(image)
        class_weights = torch.exp(log_q)

        # sample
        z_sample = bs_lib.sample_class_weights(class_weights)
        z_sample_one_hot = common_utils.get_one_hot_encoding_from_int(z_sample)

        # get losses
        image_loss = get_conditional_loss(z_sample_one_hot, image)

        mse = ((image_loss.detach() - baseline_nn(image))**2).sum()

        mse.backward()
        optimizer.step()

        avg_mse = mse / num_images

    return avg_mse

def get_nvil_baseline_nn_warmstart(baseline_nn, vae_init_file, loader, epochs,
                    outfile = '../mnist_vae_results/baseline_nn_warmstart'):

    vae = mnist_vae_lib.MovingHandwritingVAE()

    print('using vae from ', vae_init_file)
    vae.load_state_dict(torch.load(vae_init_file,
                                   map_location=lambda storage, loc: storage))

    optimizer = optim.Adam([
                {'params': baseline_nn.parameters(),
                'lr': 1e-2,
                'weight_decay': 1e-5}])

    for epoch in range(epochs):
        avg_mse = eval_nvil_baseline_nn(baseline_nn, vae.pixel_attention,
                        vae.get_loss_cond_pixel_1d,
                        loader, optimizer)

        print('epoch {}; avg_mse: {}'.format(epoch, avg_mse))


    print("writing the baseline_nn parameters to " + outfile + '\n')
    torch.save(baseline_nn.state_dict(), outfile)

##################
np.random.seed(901)
_ = torch.manual_seed(901)

data_dir = '../mnist_data/'
propn_sample = 0.1
train_set, test_set = \
    mnist_data_utils.get_moving_mnist_dataset(data_dir, propn_sample)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=64,
                 shuffle=True)


grad_estimator = bs_lib.nvil
baseline_nn = bs_lib.BaselineNN(slen = train_set[0]['image'].shape[-1])

get_nvil_baseline_nn_warmstart(baseline_nn,
                    vae_init_file = '../mnist_vae_results/moving_mnist_vae_reinforce_final',
                    loader = train_loader,
                    epochs = 100,
                    outfile = '../mnist_vae_results/baseline_nn_warmstart'):
