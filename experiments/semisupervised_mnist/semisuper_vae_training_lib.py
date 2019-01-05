import torch
from torch import optim

from itertools import cycle

import time

import numpy as np

import vae_utils

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_correct_classifier(image, true_labels, n_classes, fudge_factor = 1e-12):
    # for debugging only: returns q with mass on the correct label

    batch_size = image.shape[0]
    q = torch.zeros((batch_size, n_classes)) + fudge_factor
    seq_tensor = torch.LongTensor([i for i in range(batch_size)])
    q[seq_tensor, true_labels] = 1 - fudge_factor * (n_classes - 1)

    assert np.all((q > 0).detach().numpy())

    return torch.log(q).to(device)


def eval_semisuper_vae(vae, classifier, loader_unlabeled,
                            loader_labeled = [None],
                            train = False, optimizer = None,
                            topk = 0, use_baseline = True,
                            n_samples = 1,
                            train_labeled_only = False):

    if train:
        assert optimizer is not None
        vae.train()
        classifier.train()

    else:
        vae.eval()
        classifier.eval()

    sum_loss = 0.0
    num_images = 0.0

    for labeled_data, unlabeled_data in zip(cycle(loader_labeled), \
                                                loader_unlabeled):

        unlabeled_image = unlabeled_data['image'].to(device)

        if labeled_data is not None:
            # get labeled portion of loss

            labeled_image = labeled_data['image'].to(device)
            given_labels = labeled_data['label'].to(device)

            # get labeled loss
            labeled_loss = \
                vae_utils.get_labeled_loss(vae, labeled_image,
                                            given_labels).mean()

            # cross entropy term
            labeled_log_q = classifier.forward(labeled_image)

            cross_entropy = \
                vae_utils.get_class_label_cross_entropy(labeled_log_q,
                                        given_labels) / labeled_image.shape[0]

            num_labeled = len(loader_labeled.sampler)

        else:
            labeled_loss = 0.0
            cross_entropy = 0.0
            num_labeled = 0.0

        # run through classifier
        log_q = classifier.forward(unlabeled_image)

        if train:

            train_labeled_only_bool = 1.
            if train_labeled_only:
                n_samples = 0
                train_labeled_only_bool = 0.

            # flush gradients
            optimizer.zero_grad()

            # get unlabeled pseudoloss
            f_z = lambda z : vae_utils.get_labeled_loss(vae, unlabeled_image, z)
            unlabeled_ps_loss = 0.0
            for i in range(n_samples):
                unlabeled_ps_loss_ = rb_lib.get_raoblackwell_ps_loss(f_z, log_q,
                                        topk = topk,
                                        use_baseline = use_baseline)

                unlabeled_ps_loss += unlabeled_ps_loss_

            unlabeled_ps_loss = unlabeled_ps_loss / max(n_samples, 1)

            kl_q = torch.sum(torch.exp(log_q) * log_q)

            total_ps_loss = \
                (unlabeled_ps_loss + kl_q) * train_labeled_only_bool * \
                len(loader_unlabeled.sampler) / unlabeled_image.shape[0] + \
                (labeled_loss + cross_entropy) * num_labeled

            # backprop gradients from pseudo loss
            total_ps_loss.backward()
            optimizer.step()

        # loss at MAP value of z
        loss = \
            vae_utils.get_labeled_loss(vae, unlabeled_image,
                                torch.argmax(log_q, dim = 1)).detach().sum()

        sum_loss += loss
        num_images += unlabeled_image.shape[0]

    return sum_loss / num_images


def train_vae(vae, classifier,
                train_loader, test_loader,
                optimizer,
                loader_labeled = [None],
                train_labeled_only = False,
                topk = 0, n_samples = 1, use_baseline = True,
                epochs=10,
                save_every = 10,
                print_every = 10,
                outfile='./ss_mnist'):

    # initial losses
    init_train_loss = eval_semisuper_vae(vae, classifier, train_loader)
    init_train_accuracy = get_classification_accuracy(classifier, train_loader)
    print('init train loss: {} || init train accuracy: {}'.format(
                init_train_loss, init_train_accuracy))

    train_loss_array = [init_train_loss]
    batch_losses = [init_train_loss]
    train_accuracy_array = [init_train_accuracy]

    init_test_loss = eval_semisuper_vae(vae, classifier, test_loader)
    init_test_accuracy = get_classification_accuracy(classifier, test_loader)
    print('init test loss: {} || init test accuracy: {}'.format(
                init_test_loss, init_test_accuracy))

    test_loss_array = [init_test_loss]
    test_accuracy_array = [init_test_accuracy]

    epoch_start = 1
    for epoch in range(epoch_start, epochs+1):

        t0 = time.time()

        loss = eval_semisuper_vae(vae, classifier, train_loader,
                            loader_labeled = loader_labeled,
                            topk = topk,
                            n_samples = n_samples,
                            use_baseline = use_baseline,
                            train = True,
                            optimizer = optimizer,
                            train_labeled_only = train_labeled_only)

        elapsed = time.time() - t0
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, loss, elapsed))
        batch_losses.append(loss)
        np.save(outfile + '_batch_losses', np.array(batch_losses))

        # print stuff
        if epoch % print_every == 0:
            # save the checkpoint.
            train_loss = eval_semisuper_vae(vae, classifier, train_loader)
            test_loss = eval_semisuper_vae(vae, classifier, test_loader)

            print('train loss: {}'.format(train_loss) + \
                    ' || test loss: {}'.format(test_loss))

            train_loss_array.append(train_loss)
            test_loss_array.append(test_loss)
            np.save(outfile + '_train_losses', np.array(train_loss_array))
            np.save(outfile + '_test_losses', np.array(test_loss_array))


            train_accuracy = get_classification_accuracy(classifier, train_loader)
            test_accuracy = get_classification_accuracy(classifier, test_loader)

            print('train accuracy: {}'.format(train_accuracy) + \
                    ' || test accuracy: {}'.format(test_accuracy))

            train_accuracy_array.append(train_accuracy)
            test_accuracy_array.append(test_accuracy)
            np.save(outfile + '_train_accuracy', np.array(train_accuracy_array))
            np.save(outfile + '_test_accuracy', np.array(test_accuracy_array))

        if epoch % save_every == 0:
            outfile_epoch = outfile + 'vae_epoch' + str(epoch)
            print("writing the vae parameters to " + outfile_epoch + '\n')
            torch.save(vae.state_dict(), outfile_epoch)

            outfile_epoch = outfile + 'classifier_epoch' + str(epoch)
            print("writing the classifier parameters to " + outfile_epoch + '\n')
            torch.save(classifier.state_dict(), outfile_epoch)

    outfile_final = outfile + 'vae_final'
    print("writing the vae parameters to " + outfile_final + '\n')
    torch.save(vae.state_dict(), outfile_final)

    outfile_final = outfile + 'classifier_final'
    print("writing the classifier parameters to " + outfile_final + '\n')
    torch.save(classifier.state_dict(), outfile_final)


def get_classification_accuracy_on_batch(classifier, image, label):
    log_q = classifier(image).detach()
    z_ind = torch.argmax(log_q, dim = 1)

    return torch.mean((z_ind == label).float())


def get_classification_accuracy(classifier, loader,
                                max_images = np.inf):

    n_images = 0.0
    accuracy = 0.0

    for batch_idx, data in enumerate(loader):
        images = data['image'].to(device)
        labels = data['label'].to(device)

        accuracy += \
            get_classification_accuracy_on_batch(classifier, images, labels) * \
                    images.shape[0]

        n_images += images.shape[0]

        if n_images > max_images:
            break

    return accuracy / n_images