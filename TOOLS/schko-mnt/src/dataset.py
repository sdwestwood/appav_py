import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
#from scipy import signal
import torch
import h5py
import random
import json 
from tqdm import tqdm
from sklearn.preprocessing import normalize
from math import pi
#from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from src.eeglib.helpers import DEAPHelper
from src.eeglib import wrapper
from collections import defaultdict
from matplotlib import pyplot as plt

def get_features(helper, split_indices, freq=128, sliding_window=None, cond='sample'):
    '''
    condition = 'sample' (full eeg sample to calculate band powers), or 'band'-specific
    '''
    def window_features(input_eeg, cond):
        features = []
        if cond=='band':
            features.append(eeg.HFD())
            features.append(eeg.hjorthActivity())
            features.append(eeg.hjorthMobility())
            features.append(eeg.hjorthComplexity())
            features.append(eeg.sampEn())
            #pyeeg_hjorth = eeg.pyeeg_hjorth()
            #features.append(pyeeg_hjorth[:,0])
            #features.append(pyeeg_hjorth[:,1])
            #features.append(eeg.pyeeg_entropy())
            features = np.vstack(features)
        elif cond=='sample':
            bandpowers = eeg.pyeeg_bin_power()
            features = np.array(bandpowers).T
        return features
    
    all_feats = []
    if sliding_window:
        count=0
        for eeg in helper[::int(sliding_window*freq)]:
            if count in split_indices:
                all_feats.append(window_features(eeg,cond=cond))
            count+=1
    else:
        for eeg in helper:
            all_feats.append(window_features(eeg,cond=cond))
    return all_feats

def remove_baseline(input_eeg,baseline=[0,3],freq=128):
    return input_eeg[:,baseline[1]*freq:]

def average_baseline(input_eeg,baseline=[0, 3],freq=128,return_baseline=False):
    # average referenced
    input_eeg = input_eeg-np.mean(input_eeg,axis=0)
    # baseline corrected
    time_interval = baseline
    interval_baseline = input_eeg[:,time_interval[0]:time_interval[1]*freq]
    avg_epoch = np.mean(input_eeg[:,:interval_baseline.shape[1]], axis=1)
    for i in range(input_eeg.shape[0]):   
        input_eeg[i,:] = input_eeg[i,:] - avg_epoch[i]
    if return_baseline:
        return input_eeg
    return remove_baseline(input_eeg,baseline=[0,3],freq=128)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='deap', split_type='train', cond = 'cont', label_class = 'arousal', if_align=False, create_new_split=False, splits=(.70, .15, .15), freq = 150,
                baseline=[-0.05, 0], trial_duration = 0.2, overlap_window = 2, session_limit = 63, h5_filename = None, subset_modalities=0): # tested conditions = trial_duration = 4, overlap = 50%
        # cond = cont or binary clasification
        assert splits[0]+splits[1]+splits[2] == 1, 'split proportions do not add up to 1'
        # splits = training, valid, test
        super(Multimodal_Datasets, self).__init__()
        if data == 'deap':
            h5_path = os.path.join(dataset_path, 'deap'+'.h5')
        else:
            h5_path = os.path.join(dataset_path, h5_filename)

        h5_path = os.path.join(dataset_path, h5_filename)

        if create_new_split:
            with h5py.File(h5_path,'r') as h5_file:
                alltrials=[]
                if data == 'deap':
                    subs = []
                    sim_data_name = h5_filename[0:h5_filename.find('_EEG')]
                    for ar in list(h5_file.keys()):
                        if len(list(h5_file[ar])) != 0:
                            for tr in list(h5_file[ar].keys()):
                                subs.append(f'{ar}/{sim_data_name}/{tr}')
                        else:
                            subs.append(f'{ar}/{sim_data_name}/trial_none')
                elif data == 'nidyn':
                    subs = []
                    for ar in list(h5_file['ds']['seed'].keys()):
                        for sub in list(h5_file['ds']['seed'][ar].keys()):
                            for tr in list(h5_file['ds']['seed'][ar][sub].keys()):
                                subs.append(f'{ar}/{sub}/{tr}')
                elif data == 'eegsim':
                    subs = []
                    for sub in list(h5_file.keys()):
                        for tr in list(h5_file[sub].keys()):
                            subs.append(f'{sub}/{tr}')
                for sub in subs:
                    # chunk up the trial into blocks of e.g. 5 seconds to create samples
                    start_sample_idx = baseline[1]*freq # start after baseline period
                    total_expected_samples = session_limit*freq # end at the session limit
                    
                    if overlap_window > 0:
                        sample_dur_interval = int(overlap_window*freq)
                    else:
                        sample_dur_interval = int(trial_duration*freq)
                    
                    ts_count = 0
                    
                    if data == 'deap':
                        for start_idx in range(start_sample_idx, total_expected_samples, sample_dur_interval): # no overlap/sliding_window
                            end_idx = start_idx + int(trial_duration*freq)
                            if end_idx <= total_expected_samples:
                                alltrials.append([sub+'_'+k.rsplit('_',1)[1]+'_'+str(ts_count) for k in list(h5_file['all'][sub].keys()) if 'trial_eeg' in k])
                                ts_count += 1
                    elif data == 'nidyn':
                        alltrials.append([sub+'_'+k+'_'+str(ts_count) for k in list(h5_file['ds']['seed'][sub].keys()) if 'eeg' in k])
                    elif data == 'eegsim':
                        alltrials.append([sub+'_'+k+'_'+str(ts_count) for k in list(h5_file[sub].keys()) if 'eeg' in k])
                        
                
                alltrials = [item for sublist in alltrials for item in sublist]
                random.shuffle(alltrials)
                
                train_ids = alltrials[:int(len(alltrials)*splits[0])]
                valid_ids = alltrials[int(len(alltrials)*splits[0]):int(len(alltrials)*(splits[0]+splits[1]))] # 15% valid
                test_ids = alltrials[int(len(alltrials)*(splits[0]+splits[1])):] # 15% test
                
                train_ids_dict = defaultdict(list)
                valid_ids_dict = defaultdict(list)
                test_ids_dict = defaultdict(list)
                
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
                
                if data != 'nidyn':
                    with open(os.path.join(dataset_path, data+"_splits.json"), "w") as outfile: 
                        json.dump(splits_dict, outfile)
                else:
                    with open(os.path.join(dataset_path, h5_filename[:-3]+"_splits.json"), "w") as outfile: 
                        json.dump(splits_dict, outfile)

        else:
            try:
                if data == 'deap':
                    with open(os.path.join(dataset_path, data+"_splits.json")) as json_file:
                        splits_dict = json.load(json_file)
                else:
                    with open(os.path.join(dataset_path, h5_filename[:-3]+"_splits.json")) as json_file:
                        splits_dict = json.load(json_file)
            except FileNotFoundError:
                with open('../data/deap_splits.json') as json_file:
                    splits_dict = json.load(json_file)
                h5_path = '../data/deap.h5'
        
        # These are torch tensors
        self.split_type = split_type
        self.splits_dict = splits_dict
        # self.ecg_channels = ['plethysmograph']
        # self.eye_channels = ['GSR'] 
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
        if self.data == 'deap':
            self.n_modalities = 4 # vision/ text/ audio
        elif subset_modalities > 0:
            self.n_modalities = subset_modalities # two modalities typically
        elif self.data == 'nidyn' and subset_modalities == 0:
            self.n_modalities = 3
        if data == 'deap':
            self._add_data_infos(h5_path)
            self.eeg_channels = ['TP7', 'CP5', 'CP3', 'CP1', 'CPz','CP2', 'CP4', 'CP6', 'TP8'
                                 'P9','P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                                 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
        elif data == 'nidyn':
            self._add_data_infos_nidyn(h5_path)
            self.eeg_channels = np.arange(start=1, stop=21)
        elif data == 'eegsim':
            self._add_data_infos_eegsim(h5_path)
        
        self.eeg = torch.tensor(np.array(self.eeg, dtype=np.float32)).cpu().detach()
        self.ecg = torch.tensor(np.array(self.ecg, dtype=np.float32)).cpu().detach() 
        self.features = torch.tensor(np.array(self.features, dtype=np.float32)).cpu().detach()
        self.eye = torch.tensor(np.array(self.eye, dtype=np.float32)).cpu().detach()
        self.cond = cond
        self.labels = torch.tensor(np.array(self.labels, dtype=np.float32)).cpu().detach()
        # Note: this is STILL an numpy array
        # self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.meta = None # nto sure what this is..
        self.nidyn_filename = h5_filename
        
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
                #eeg_data = (eeg_data - (-0.00002))/(0.00007-(-0.00002)) # manual normaliztion
                # eeg_data = torch.nn.functional.normalize(torch.tensor(eeg_data)).numpy()
                
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