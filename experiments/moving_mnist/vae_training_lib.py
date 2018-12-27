import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_vae(vae, loader, \
                optimizer = None,
                train = False,
                set_true_loc = False,
                topk = 0,
                use_baseline = True,
                n_samples = 1):
    if train:
        vae.train()
        assert optimizer is not None
    else:
        n_samples = 0
        vae.eval()

    avg_loss = 0.0

    num_images = len(loader.dataset)

    for batch_idx, data in enumerate(loader):

        if optimizer is not None:
            optimizer.zero_grad()

        image = data['image'].to(device)
        if set_true_loc:
            true_pixel_2d = data['pixel_2d'].to(device)
            n_samples = 1
            use_baseline = False
            topk = 0
        else:
            true_pixel_2d = None

        pm_loss, loss = vae.get_rb_loss(image,
                                        topk = topk,
                                        use_baseline = use_baseline,
                                        n_samples = n_samples,
                                        true_pixel_2d = true_pixel_2d)

        if train:
            pm_loss.backward()
            optimizer.step()

        avg_loss += loss.data  / num_images

    return avg_loss

def train_vae(vae, train_loader, test_loader, optimizer,
                    set_true_loc = False,
                    topk = 0,
                    use_baseline = True,
                    n_samples = 1,
                    outfile = './mnist_vae_semisupervised',
                    n_epoch = 200, print_every = 10, save_every = 20):

    # get losses
    train_loss = eval_vae(vae, train_loader, train = False,
                            set_true_loc = set_true_loc)
    test_loss = eval_vae(vae, test_loader, train = False,
                            set_true_loc = set_true_loc)

    print('  * init train recon loss: {:.10g};'.format(train_loss))
    print('  * init test recon loss: {:.10g};'.format(test_loss))

    for epoch in range(1, n_epoch + 1):
        start_time = timeit.default_timer()

        loss = eval_vae(vae, train_loader,
                                optimizer = optimizer,
                                train = True,
                                set_true_loc = set_true_loc,
                                topk = topk,
                                use_baseline = use_baseline,
                                n_samples = n_samples)

        elapsed = timeit.default_timer() - start_time
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))

        if epoch % print_every == 0:
            train_loss = eval_vae(vae, train_loader, train = False,
                                    set_true_loc = set_true_loc)
            test_loss = eval_vae(vae, test_loader, train = False,
                                    set_true_loc = set_true_loc)

            print('  * train recon loss: {:.10g};'.format(train_loss))
            print('  * test recon loss: {:.10g};'.format(test_loss))

        if epoch % save_every == 0:
            outfile_every = outfile + '_epoch' + str(epoch)
            print("writing the parameters to " + outfile_every + '\n')
            torch.save(vae.state_dict(), outfile_every)

    outfile_final = outfile + '_final'
    print("writing the parameters to " + outfile_final + '\n')
    torch.save(vae.state_dict(), outfile_final)
