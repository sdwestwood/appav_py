import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch
import h5py
import random
from collections import defaultdict
from tqdm import tqdm
import json



if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset): # I have set split default to True, adjusted default data names to 'sub-001'
    def __init__(self, dataset_path, data='sub-001', split_type='train', cond = 'cont', label_class = 'arousal', if_align=False, create_new_split=True, splits=(.70, .15, .15), freq = 150,
                baseline=[-0.05, 0], trial_duration = 0.2, overlap_window = 2, session_limit = 63, h5_filename = 'sub-001.h5', subset_modalities = 2): # tested conditions = trial_duration = 4, overlap = 50%
        # cond = cont or binary clasification
        assert splits[0]+splits[1]+splits[2] == 1, 'split proportions do not add up to 1'
        # splits = training, valid, test
        super(Multimodal_Datasets, self).__init__()

        # set filepath to the h5 file
        h5_path = os.path.join(dataset_path, h5_filename) # !! adjusted to my location

        if create_new_split:
            with h5py.File(h5_path,'r') as h5_file:
                alltrials=[]
                subs = []              
                
                # populate subs with trial number 'tr' for each subject
                # tr is the trial number key (each one contains eeg and pupil data)
                # !! I have removed the dataset-specific options 
                for sub in list(h5_file.keys()):
                    for tr in list(h5_file[sub].keys()):
                        subs.append(f'{sub}/{tr}')

                # for each subject, populate alltrials with the list of trial numbers for EEG only
                for sub in subs:                
                    ts_count = 0
                    # !! removed code relating to deap and nidyn datasets for now (unsure what format is needed)
                    alltrials.append([sub+'_'+k+'_'+str(ts_count) for k in list(h5_file[sub].keys()) if 'eeg' in k])
                             
                # this code flattens the shape of alltrials into list? then shuffles   
                alltrials = [item for sublist in alltrials for item in sublist] 
                random.shuffle(alltrials)
                
                # assign splits to shuffled alltrials data 
                train_ids = alltrials[:int(len(alltrials)*splits[0])]
                valid_ids = alltrials[int(len(alltrials)*splits[0]):int(len(alltrials)*(splits[0]+splits[1]))] # 15% valid
                test_ids = alltrials[int(len(alltrials)*(splits[0]+splits[1])):] # 15% test
                
                train_ids_dict = defaultdict(list)
                valid_ids_dict = defaultdict(list)
                test_ids_dict = defaultdict(list)
                
                # populate dicts with original trial idx for shuffled splits
                for ids in train_ids:
                    train_ids_dict[ids.rsplit('_',1)[0]].append(int(ids.rsplit('_',1)[1]))
                for ids in valid_ids:
                    valid_ids_dict[ids.rsplit('_',1)[0]].append(int(ids.rsplit('_',1)[1]))
                for ids in test_ids:
                    test_ids_dict[ids.rsplit('_',1)[0]].append(int(ids.rsplit('_',1)[1]))
                
                # create dict of the other 3 dicts
                splits_dict = {'train': train_ids_dict, 'valid': valid_ids_dict, 'test': test_ids_dict}
                
                # sort windows within subject and trial, in order
                # this doesn't do anything if not doing the windows right?
                for split in splits_dict:
                    for sub in splits_dict[split]:
                        indices = splits_dict[split][sub]
                        splits_dict[split][sub] = sorted(indices)
                
                # !! removed option for nidyn dataset as we are using h5 format
                with open(os.path.join(dataset_path, h5_filename[:-3]+"_splits.json"), "w") as outfile: 
                    json.dump(splits_dict, outfile)

        else: # !! again removed extra option leaving only generic h5 option for now
            with open(os.path.join(dataset_path, h5_filename[:-3]+"_splits.json")) as json_file:
                splits_dict = json.load(json_file)
                    

        # These are torch tensors
        self.split_type = split_type
        self.splits_dict = splits_dict
        self.label_type = cond
        self.freq = freq
        self.baseline = baseline
        self.bandpass = [0.5, 40] # bandpass freq limits for EEG
        self.trial_duration = trial_duration # how long the trials are so we can split the data (s)
        self.session_limit = session_limit # how long each data collection session lasts to chunk while creating samples (s)
        # run through h5 file and add
        self.data_cache = {}
        self.data_info = []
        self.labels = []
        self.acc = []
        self.eeg = []
        self.features = []
        self.eye = []
        self.plot_data = False
        self.label_class = 'both'
        self.overlap_window = overlap_window
        self.subset_modalities = subset_modalities
        self.data = data
        self.n_modalities = subset_modalities # !! removed dataset-specific n modalities
        ### 
        self._add_data_infos(h5_path) # !! removed data-specific options - will need to edit the function as well 
        ###
        self.eeg_channels = np.arange(start=1, stop=65) # !! I am guessing the format that will work for my channels        
        self.eeg = torch.tensor(np.array(self.eeg, dtype=np.float32)).cpu().detach()
        self.features = torch.tensor(np.array(self.features, dtype=np.float32)).cpu().detach()
        self.eye = torch.tensor(np.array(self.eye, dtype=np.float32)).cpu().detach()
        self.cond = cond
        self.labels = torch.tensor(np.array(self.labels, dtype=np.float32)).cpu().detach()
        self.acc = torch.tensor(np.array(self.acc, dtype=np.float32)).cpu().detach()
        # Note: this is STILL an numpy array
        # self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.meta = None # not sure what this is..
        self.h5_filename = h5_filename # !! changed from self.nidyn_filename 
        
    def _add_data_infos(self, file_path):
        print('Loading',file_path)

        convert_cond = lambda x: float(x == 'high')
        with h5py.File(file_path,'r') as h5_file:
            # Walk through all groups, extracting datasets
            for split_id in tqdm(self.splits_dict[self.split_type]):
                subid, trialno = split_id.split('_')
                split_indices = self.splits_dict[self.split_type][split_id]
                pp_ds = h5_file[str(subid)]   
                all_data = pp_ds
                eeg_data = np.array(all_data['eeg'])
                
                means = eeg_data.mean(0)
                stds = eeg_data.std(0)
                #eeg_data = (eeg_data - means) / stds

                self.eeg.append(eeg_data[:,:])
                
                # eye 
                eye_data = np.array(all_data['pupil'])
                means = eye_data.mean()
                stds = eye_data.std()
                #eye_data = (eye_data - (-0.00002))/(0.00007-(-0.00002)) # manual normaliztion
                #eye_data = (eye_data - means) / stds
                # eye_data = torch.nn.functional.normalize(torch.tensor(eye_data),p=.05).numpy()
                #scaler = MinMaxScaler()
                #eye_data = scaler.fit_transform(eye_data.T).T
                    
                # save trial data
                self.labels.append(np.array(all_data['cond'])) # punishment == 0, reward == 1
                self.acc.append(np.array(all_data['acc'])) # incorrect == 0, correct == 1
                self.eye.append(eye_data)
                
    def get_n_modalities(self):
        return self.n_modalities
    
    def get_seq_len(self):
        # !! removed if statement for nidyn and copied eegsim format, removed ecg
        if self.n_modalities == 2:
            return self.eeg.shape[1], self.eye.shape[1]
        return self.eeg.shape[1], self.eye.shape[1], self.features.shape[1]
    
    def get_mod_type(self):
        # TODO: get types of modalities (time series or features to know how to treat the initial conv)
        return None
    
    def get_dim(self):
        # !! removed if statement for nidyn and copied eegsim format, removed ecg
        if self.n_modalities == 2:
            return self.eeg.shape[2], self.eye.shape[2]
        return self.eeg.shape[2], self.eye.shape[2]
    
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.n_modalities == 2:
            X = (index, self.eeg[index], self.eye[index])
            Y = torch.tensor(self.labels[index], dtype=torch.long)
            return X, Y, META
        X = (index, self.eeg[index], self.eye[index], self.features[index], self.ecg[index])
        if self.label_class == 'arousal':
            lab_idx = 2
        elif self.label_class== 'valence':
            lab_idx = 1
        else:
            assert "Label class must be valence or arousal"
        Y = self.labels[index][lab_idx]     
        if self.cond == 'binary' and self.labels.dtype==torch.float32: # if we haven't converted the labels yet, do it on the fly
            if Y > 5:
                Y = torch.tensor(1, dtype=torch.long)
            else:
                Y = torch.tensor(0, dtype=torch.long)
        return X, Y, META