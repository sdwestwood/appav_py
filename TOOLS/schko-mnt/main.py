path_to_base_package = '/home/jupyter/causalmnt_sharath'
import sys
# setting path
sys.path.append(f"{path_to_base_package}")

import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from torch.utils.data import ConcatDataset
import torch.nn.functional as f
from sklearn.preprocessing import normalize as sknorm
import numpy as np
import ast
torch.cuda.empty_cache()


#if __name__ == '__main__':
#    """
#    This file is intended to be used to adapt Pawan's data for use with the MuLT model.
#    """
#    main()

#def main(args):
parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')
# Label type set to binary (target vs distractor)
parser.add_argument('--label_type', type=str, default='binary',
                    help='type of deap label')
# TODO: Check if default is target or distractor
parser.add_argument('--target_distractor', type=str, default='target',
                    help='type of class')
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
parser.add_argument('--dataset', type=str, default='nidyn',
                    help='dataset to use (default: nidyn)')
parser.add_argument('--dataset_type', type=str, default='nidyn',
                    help='extra metadata on dataset for training storage (default: nidyn)')
parser.add_argument('--data_path', type=str, default='../data/nidyn_data',
                    help='path for storing the dataset')
parser.add_argument('--h5_filename', type=str, default='NiDyN_ND1_High.h5',
                    help='filename to use')

# Dropouts
parser.add_argument('--attn_dropout_l', type=float, default=0.1,
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
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (original: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=10,
                    help='when to decay learning rate (original: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=1,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
#args = parser.parse_args(args)
args = parser.parse_args()
torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
args.subset_channels = ast.literal_eval(args.subset_channels)

# Remove honly as only 3 modalities used
valid_partial_mode = args.lonly + args.vonly + args.aonly 

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly  = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'deap': 1,
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8, 
    'pawan': 1
}

if args.label_type=='binary':
    output_dim_dict['deap'] = 2

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

if args.label_type=='binary':
    criterion_dict['deap'] = 'BCEWithLogitsLoss'
    criterion_dict['nidyn'] = 'BCEWithLogitsLoss'
    criterion_dict['eegsim'] = 'BCEWithLogitsLoss'

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

if 'deap' in dataset: # pre-process deap data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feats = train_data.eye.numpy()
    # normalize features
    feats = sknorm(feats.reshape(-1,feats.shape[1]*feats.shape[2])).reshape(-1, feats.shape[1], feats.shape[2])
    for i in range(feats.shape[1]):
        feats[:, i, :] = scaler.fit_transform(feats[:, i, :]) 
    train_data.eye = torch.from_numpy(feats)

    feats = valid_data.eeg.numpy()
    feats = sknorm(feats.reshape(-1,feats.shape[1]*feats.shape[2])).reshape(-1, feats.shape[1], feats.shape[2])
    for i in range(feats.shape[1]):
        feats[:, i, :] = scaler.fit_transform(feats[:, i, :]) 
    valid_data.eeg = torch.from_numpy(feats)

    feats = test_data.head.numpy()
    feats = sknorm(feats.reshape(-1,feats.shape[1]*feats.shape[2])).reshape(-1, feats.shape[1], feats.shape[2])
    for i in range(feats.shape[1]):
        feats[:, i, :] = scaler.fit_transform(feats[:, i, :]) 
    test_data.head = torch.from_numpy(feats)

if len(args.subset_channels)>0 and 'nidyn' in dataset:
    eeg_data = train_data.eeg.numpy()
    eeg_data = np.take(eeg_data,args.subset_channels,axis=2)
    train_data.eeg = torch.from_numpy(eeg_data)
    
    eeg_data = valid_data.eeg.numpy()
    eeg_data = np.take(eeg_data,args.subset_channels,axis=2)
    valid_data.eeg = torch.from_numpy(eeg_data)
    
    eeg_data = test_data.eeg.numpy()
    eeg_data = np.take(eeg_data,args.subset_channels,axis=2)
    test_data.eeg = torch.from_numpy(eeg_data)

if args.subset_modalities>0 and 'nidyn' in dataset:
    train_data.n_modalities = args.subset_modalities
    valid_data.n_modalities = args.subset_modalities
    test_data.n_modalities = args.subset_modalities

# set conditon in case we want binary transformations of the data
train_data.label_class = args.target_distractor
valid_data.label_class = args.target_distractor
test_data.label_class = args.target_distractor

train_data.cond = args.label_type
valid_data.cond = args.label_type
test_data.cond = args.label_type

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

if 'deap' in dataset:
    args.dataset_type = dataset
    dataset = 'deap'


print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.modality_count = len(train_data.get_dim())
if hyp_params.modality_count == 2:
    hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim() # assumes one time series, one feature modality
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
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
hyp_params.feat_kernel_size = 1
hyp_params.feat_stride = 1
if 'deap' in dataset or 'nidyn' in dataset:
    hyp_params.raw_kernel_size = int(hyp_params.l_len/12) # int(hyp_params.l_len/12)
    hyp_params.raw_stride=int(hyp_params.l_len/250) # int(hyp_params.l_len/75)
else:
    hyp_params.raw_kernel_size = 1 # int(hyp_params.l_len/12)
    hyp_params.raw_stride= 1 # int(hyp_params.l_len/75)
#if '_full' in hyp_params.dataset_type: # this is due to memory limitations
#    args.raw_kernel_size = 640
#    args.raw_stride=100
##else:
#    args.raw_kernel_size = 64 # was 64 with .72 acc ar
#    args.raw_stride = 10 # was 10 with .72 acc ar
hyp_params.valence_arousal = 'target'
test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
