import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
# additional imorts defined by sharath
from torch.utils.data import ConcatDataset
import torch.nn.functional as f
from sklearn.preprocessing import normalize as sknorm
import numpy as np
import ast
#torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Appav Accuracy Prediction')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')
# Label type set to binary (temporary, ideally want 4 classes)
parser.add_argument('--label_type', type=str, default='binary',
                    help='type of label')
parser.add_argument('--overlap_window', type=float, default=0,
                    help='seconds of data to overlap in sliding window trial to trial')
parser.add_argument('--trial_duration', type=int, default=4,
                    help='how long each trial is')
parser.add_argument('--subset_channels', type=str, default='[]',
                    help='which channels to subset')
parser.add_argument('--subset_modalities', type=int, default=0,
                    help='which modalities to subset, predefined order')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='sub-001',
                    help='dataset to use (default: sub-001)')
parser.add_argument('--dataset_type', type=str, default='appav',
                    help='extra metadata on dataset for training storage (default: appav)')
parser.add_argument('--data_path', type=str, default='data/appav',
                    help='path for storing the dataset')
parser.add_argument('--h5_filename', type=str, default='sub-001.h5',
                    help='filename to use')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--shared_dim', type=int, default=20,
                    help='shared dimension size') # minimum 4, typically 20
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'test': 1,
}

criterion_dict = {
    'appav': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

#  text, audio, features, head, = ['l', 'a', 'v', 'h'] - IDK what this means but seems to be the labeling
# v seems to be feature matrix - I think this is features of the EEG? Like the power and stuff?
# h seems to be EEG?

# think I need to choose - a can be EEG time series, v can pupil - can add feature matrix later


hyp_params = args
hyp_params.modality_count = len(train_data.get_dim())
if hyp_params.modality_count == 2:
    hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim() # for me this is 2 times series i guess - EEG and eye
    hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
    hyp_params.l_len = hyp_params.a_len # temp fix
elif hyp_params.modality_count == 3:
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
elif hyp_params.modality_count == 4:
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v, hyp_params.orig_d_h = train_data.get_dim()
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len, hyp_params.h_len = train_data.get_seq_len()

hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset_type = args.dataset_type
hyp_params.dataset = args.dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
hyp_params.feat_kernel_size = 1
hyp_params.feat_stride = 1
hyp_params.raw_kernel_size = 1 # int(hyp_params.l_len/12)
hyp_params.raw_stride= 1 # int(hyp_params.l_len/75)
#if '_full' in hyp_params.dataset_type: # this is due to memory limitations
#    args.raw_kernel_size = 640
#    args.raw_stride=100
##else:
#    args.raw_kernel_size = 64 # was 64 with .72 acc ar
#    args.raw_stride = 10 # was 10 with .72 acc ar

test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

