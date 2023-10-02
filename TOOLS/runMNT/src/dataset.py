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
                baseline=[-0.05, 0], trial_duration = 0.2, overlap_window = 2, session_limit = 63, h5_filename = 'sub-001.h5', subset_modalities=0): # tested conditions = trial_duration = 4, overlap = 50%
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
                # !! I have adjusted the sim data name so that it just appends sim to sub-001 (for now)
                sim_data_name = h5_filename[0:h5_filename.find('.')]+'_sim' 
                
                # populate subs with trial number 'tr' for each top-level key
                # ar is the list of top level keys - is this just name of data? 
                # tr is the trial number key (each one contains eeg and pupil data)
                for sub in list(h5_file.keys()):
                    for tr in list(h5_file[sub].keys()):
                        subs.append(f'{sub}/{tr}')

                # populate alltrials with whatever info is in k for each item in sub
                for sub in subs:                
                    ts_count = 0
                    # !! removed code relating to deap and nidyn datasets for now (unsure what format is needed)
                    alltrials.append([sub+'_'+k+'_'+str(ts_count) for k in list(h5_file[sub].keys()) if 'eeg' in k])
                             
                # this code flattens the shape of alltrials? then shuffles   
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
                
                splits_dict = {'train': train_ids_dict, 'valid': valid_ids_dict, 'test': test_ids_dict}
                
                # sort windows within subject and trial, in order
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
        self.bandpass = [1, 55] # bandpass freq limits for EEG
        self.trial_duration = trial_duration # how long the trials are so we can split the data (s)
        self.session_limit = session_limit # how long each data collection session lasts to chunk while creating samples (s)
        # run through h5 file and add
        self.data_cache = {}
        self.data_info = []
        self.labels = []
        self.eeg = []
        self.ecg = [] # this is an option, 4th modality (fixed time series for now)
        self.features = []
        self.eye = []
        self.plot_data = False
        self.label_class = 'both'
        self.overlap_window = overlap_window
        self.subset_modalities = subset_modalities
        self.data = data
        self.n_modalities = subset_modalities # !! removed dataset-specific n modalities
        self._add_data_infos(h5_path) # !! removed data-specific options - will need to edit the function as well
        self.eeg_channels = np.arange(start=1, stop=65) # !! I am guessing the format that will work for my channels        
        self.eeg = torch.tensor(np.array(self.eeg, dtype=np.float32)).cpu().detach()
        self.ecg = torch.tensor(np.array(self.ecg, dtype=np.float32)).cpu().detach() 
        self.features = torch.tensor(np.array(self.features, dtype=np.float32)).cpu().detach()
        self.eye = torch.tensor(np.array(self.eye, dtype=np.float32)).cpu().detach()
        self.cond = cond
        self.labels = torch.tensor(np.array(self.labels, dtype=np.float32)).cpu().detach()
        # Note: this is STILL an numpy array
        # self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.meta = None # not sure what this is..
        self.h5_filename = h5_filename # !! changed from self.nidyn_filename 
        
    def _add_data_infos_eegsim(self, file_path):
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
                self.labels.append(np.array(convert_cond(subid.split('/')[0]))) # low == 0, high == 1
                self.eye.append(eye_data)
                
    def _add_data_infos(self, file_path):
        print('Loading',file_path)
        
        with h5py.File(file_path,'r') as h5_file:
            # Walk through all groups, extracting datasets
            for split_id in tqdm(self.splits_dict[self.split_type]):
                subid, trialno = split_id.split('_')
                split_indices = self.splits_dict[self.split_type][split_id]
                pp_ds = h5_file['all'][str(subid)]                
                all_data = pp_ds['trial_eeg_'+trialno][:]
                
                # correction for nan or inf values
                all_data = np.nan_to_num(all_data)
                channels = eval(pp_ds.attrs['channels'])
                
                # EEG processing
                eeg_idx = [i for i, e in enumerate(channels) if e in self.eeg_channels]
                eeg_data = np.take(all_data, eeg_idx, 0)
                
                # feature extraction for non-timedomain modality
                proc_eeg = DEAPHelper(average_baseline(eeg_data.copy()),names=self.eeg_channels,
                                      sampleRate=self.freq, windowSize=int(self.freq*self.trial_duration), normalize=True)
                if self.overlap_window > 0:
                    all_features = get_features(proc_eeg, split_indices, sliding_window=self.overlap_window, cond='sample')
                else:
                    all_features = get_features(proc_eeg, split_indices, cond='sample')
                helpers = [] # holds theta alpha, beta, gamma waves
                helpers.append(DEAPHelper(average_baseline(eeg_data.copy()),names=self.eeg_channels,
                                          sampleRate=self.freq, windowSize=int(self.freq*self.trial_duration), normalize=True, highpass=3, lowpass=8))
                helpers.append(DEAPHelper(average_baseline(eeg_data.copy()),names=self.eeg_channels,
                                          sampleRate=self.freq, windowSize=int(self.freq*self.trial_duration), normalize=True, highpass=8, lowpass=13))
                helpers.append(DEAPHelper(average_baseline(eeg_data.copy()),names=self.eeg_channels,
                                          sampleRate=self.freq, windowSize=int(self.freq*self.trial_duration), normalize=True, highpass=13, lowpass=30))
                helpers.append(DEAPHelper(average_baseline(eeg_data.copy()),names=self.eeg_channels,
                                          sampleRate=self.freq, windowSize=int(self.freq*self.trial_duration), normalize=True, highpass=30))
                all_features_cp = all_features
                
                for helper in helpers:
                    if self.overlap_window > 0:
                        features = get_features(helper, split_indices, sliding_window=self.overlap_window, cond='band')
                    else:
                        features = get_features(helper, split_indices, cond='band')
                    all_features = all_features_cp.copy()
                    c_count = 0
                    for window,new_feat in zip(all_features, features): # each window, add on to existing array
                        all_features_cp[c_count] = np.concatenate((window,new_feat),axis=0)
                        c_count += 1
                all_features_cp = np.array(all_features_cp)
                # normalized EEG data, can optionally ICA here
                proc_eeg = DEAPHelper(average_baseline(eeg_data.copy(),return_baseline=True),names=self.eeg_channels,
                                      sampleRate=self.freq, normalize=True)
                eeg_data = proc_eeg.data
                
                
                ecg_idx = [i for i, e in enumerate(channels) if e in self.ecg_channels]
                eye_idx = [i for i, e in enumerate(channels) if e in self.eye_channels]
                
                # ecg, eye and head processing = min-max scale, phase of baseline period can bias corrections
                ecg_data = np.take(all_data, ecg_idx, 0)
                scaler = MinMaxScaler()
                ecg_data = scaler.fit_transform(ecg_data.T).T

                # gsr
                gsr_data = np.take(all_data, eye_idx, 0)
                scaler = MinMaxScaler()
                gsr_data = scaler.fit_transform(gsr_data.T).T
                
                # chunk up the trial into blocks of e.g. 5 seconds to create samples
                start_sample_idx = self.baseline[1]*self.freq # start after baseline period
                total_expected_samples = self.session_limit*self.freq # end at the session limit
                
                if self.overlap_window > 0:
                    sample_dur_interval = int(self.overlap_window*self.freq)
                else:
                    sample_dur_interval = int(self.trial_duration*self.freq)
                ts_count = 0
                valid_samples = 0
                for start_idx in range(start_sample_idx, total_expected_samples, sample_dur_interval): # no overlap/sliding_window
                    end_idx = start_idx + int(self.trial_duration*self.freq)
                    if ts_count in split_indices and end_idx <= total_expected_samples: # make sure all data is same, expected size
                        self.labels.append(pp_ds['labels_'+trialno][:]) # continuous
                        self.eeg.append(eeg_data[:,start_idx:end_idx].T)
                        self.eye.append(gsr_data[:,start_idx:end_idx].T)
                        self.ecg.append(ecg_data[:,start_idx:end_idx].T)
                        valid_samples += 1
                    ts_count += 1
                self.features.extend(all_features_cp) # use the extracted fractal, hjorth, band power, sample entropy features as another modality
                assert len(split_indices) == all_features_cp.shape[0], "the expected number of split indices does not match the extracted features"
                assert len(split_indices) == valid_samples, 'the expected number of split indices does not match the number of time series samples'
                assert len(self.eye) == len(self.features), 'the total number of extracted features does not match the number of time series samples'
                if np.isnan(self.ecg[-1]).any() or np.isnan(self.features[-1]).any() or np.isnan(self.eye[-1]).any():
                    raise ValueError('There was an NaN value found and we havent had to handle it yet.')

                    
    def _add_data_infos_nidyn(self, file_path):
        print('Loading',file_path)
        
        with h5py.File(file_path,'r') as h5_file:
            # Walk through all groups, extracting datasets
            for split_id in tqdm(self.splits_dict[self.split_type]):
                subid, trialno = split_id.split('_')
                split_indices = self.splits_dict[self.split_type][split_id]
                pp_ds = h5_file['ds']['seed'][str(subid)]   
                all_data = pp_ds
                if self.n_modalities > 2:
                    eeg_data = np.array(all_data['eeg'])
                    self.eeg.append(eeg_data[:,:])
                
                # ecg, eye processing = min-max scale, phase of baseline period can bias corrections
                ecg_data = np.array(all_data['hr'])
                #scaler = MinMaxScaler()
                #ecg_data = scaler.fit_transform(ecg_data.T).T

                # eye 
                eye_data = np.array(all_data['pupil'])
                #scaler = MinMaxScaler()
                #eye_data = scaler.fit_transform(eye_data.T).T
                    
                # save trial data
                self.labels.append(np.array(all_data['label']))
                self.eye.append(eye_data)
                self.features.append(list(ecg_data))

                if np.isnan(self.eye[-1]).any() or np.isnan(self.features[-1]).any():
                    raise ValueError('There was an NaN value found and we havent had to handle it yet.')
                    
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        if self.data == 'nidyn':
            if self.n_modalities > 2:
                return self.eeg.shape[1], self.eye.shape[1], self.features.shape[1]
            else:
                return self.eye.shape[1], self.features.shape[1]
        elif self.data == 'eegsim':
            if self.n_modalities == 2:
                return self.eeg.shape[1], self.eye.shape[1]
        return self.eeg.shape[1], self.eye.shape[1], self.features.shape[1], self.ecg.shape[1]
    def get_mod_type(self):
        # TODO: get types of modalities (time series or features to know how to treat the initial conv)
        return None
    def get_dim(self):
        if self.data == 'nidyn':
            if self.n_modalities > 2:
                return self.eeg.shape[2], self.eye.shape[2], self.features.shape[2]
            else:
                return self.eye.shape[2], self.features.shape[2]
        elif self.data == 'eegsim':
            if self.n_modalities == 2:
                return self.eeg.shape[2], self.eye.shape[2]
        return self.eeg.shape[2], self.eye.shape[2], self.features.shape[2], self.ecg.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'nidyn':
            if self.n_modalities == 3:
                X = (index, self.eeg[index], self.eye[index], self.features[index])
            elif self.n_modalities == 2:
                X = (index, self.eye[index], self.features[index])
            Y = torch.tensor(self.labels[index], dtype=torch.long) - 1
            return X, Y, META
        elif self.data == 'eegsim' and self.n_modalities == 2:
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