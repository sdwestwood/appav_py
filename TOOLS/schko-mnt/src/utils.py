import torch
import os
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    if dataset == 'deap':
        data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}_original.dt'
    else:
        data_path = os.path.join(args.data_path, args.h5_filename[:-3]) + f'_{split}_{alignment}_original.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        if dataset == 'deap':
            data = Multimodal_Datasets(args.data_path, dataset, split, args.label_type,args.valence_arousal,trial_duration=args.trial_duration,overlap_window = args.overlap_window)
        elif dataset == 'nidyn':
            data = Multimodal_Datasets(args.data_path, dataset, split, args.label_type,args.target_distractor,trial_duration=args.trial_duration,overlap_window = args.overlap_window, freq=128, baseline=[0, 3], session_limit=63, h5_filename=args.h5_filename, create_new_split=True, subset_modalities=args.subset_modalities)
        elif dataset == 'eegsim':
            data = Multimodal_Datasets(args.data_path, dataset, split, args.label_type,args.target_distractor,trial_duration=args.trial_duration,overlap_window = args.overlap_window, freq=128, baseline=[0, 3], session_limit=63, h5_filename=args.h5_filename, create_new_split=True, subset_modalities=args.subset_modalities)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model
