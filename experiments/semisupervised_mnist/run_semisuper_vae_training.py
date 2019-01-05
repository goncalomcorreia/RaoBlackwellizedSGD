import argparse
import os

import torch

import torch.optim as optim

from copy import deepcopy

import mnist_data_lib
import mnist_vae_lib

import semisuper_vae_training_lib as ss_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training parameters
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--weight_decay', type = float, default = 1e-5)
parser.add_argument('--learning_rate', type = float, default = 0.001)

parser.add_argument('--topk', type = int, default = 0)
parser.add_argument('--n_samples', type = int, default = 1)

# whether to only train on labeled data
parser.add_argument('--train_labeled_only',
                    type=distutils.util.strtobool, default='False')

# saving vae
parser.add_argument('--outdir', type = str,
                    default='./', help = 'directory for saving encoder and decoder')
parser.add_argument('--outfilename', type = str,
                    default='moving_mnist_vae',
                    help = 'filename for saving the encoder and decoder')
parser.add_argument('--save_every', type = int, default = 50,
                    help='save encoder ever how ___ epochs (default = 50)')
parser.add_argument('--print_every', type = int, default = 10)

# Whether to just work with subset of data
parser.add_argument('--propn_sample', type = float,
                    help='proportion of dataset to use',
                    default = 1.0)

# warm start parameters
parser.add_argument('--use_vae_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae')
parser.add_argument('--vae_init_file',
                    type=str,
                    help='file to initialize the mnist vae')
parser.add_argument('--use_classifier_init',
                    type=distutils.util.strtobool, default='False',
                    help='whether to initialize the mnist vae')
parser.add_argument('--classifier_init_file',
                    type=str,
                    help='file to initialize the classifier vae')

# Other params
parser.add_argument('--seed', type=int, default=4254,
                    help='random seed')

args = parser.parse_args()

assert os.path.exists(args.outdir)

np.random.seed(args.seed)
_ = torch.manual_seed(args.seed)

# get data
train_set_labeled, train_set_unlabeled, test_set = \
    mnist_data_lib.get_mnist_dataset_semisupervised(propn_sample=args.propn_sample)

train_loader_labeled = torch.utils.data.DataLoader(
                 dataset=train_set_labeled,
                 batch_size=args.batch_size,
                 shuffle=True)

if args.train_labeled_only:
    train_loader_unlabeled = deepcopy(train_loader_labeled)
else:
    train_loader_unlabeled = torch.utils.data.DataLoader(
                     dataset=train_set_unlabeled,
                     batch_size=args.batch_size,
                     shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                shuffle=False)

# setup VAE
slen = train_set_labeled[0]['image'].shape[0]

vae, classifier = mnist_vae_lib.get_mnist_vae_and_classifier(
                        latent_dim = 8,
                        n_classes = 10,
                        slen = slen)

# get warm starts
if args.use_vae_init:
    assert os.path.isfile(init_vae_file)
    vae.load_state_dict(torch.load(args.vae_init_file,
                        map_location=lambda storage, loc: storage))

if args.use_classifier_init:
    assert os.path.isfile(init_classifier_file)
    classifier.load_state_dict(torch.load(args.classifier_init_file,
                        map_location=lambda storage, loc: storage))
                        
vae.to(device)
classifier.to(device)


# set up optimizer
optimizer = optim.Adam([
                {'params': classifier.parameters(), 'lr': args.learning_rate}, #1e-3},
                {'params': vae.parameters(), 'lr': args.learning_rate}],
                weight_decay=args.weight_decay)

# train!
outifle = args.outdir + args.outfilename
ss_lib.train_vae(vae, classifier,
                train_loader_unlabeled,
                test_loader,
                optimizer,
                train_loader_labeled,
                topk = args.topk,
                n_samples = args.n_samples,
                use_baseline = True,
                epochs=args.epochs,
                outfile = outfile,
                save_every = args.save_every,
                print_every = args.print_every,
                train_labeled_only = args.train_labeled_only)