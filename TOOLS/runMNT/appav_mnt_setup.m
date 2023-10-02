%% Convert Pupil and EEG Data to h5 format %%
clear all
close all

%% Set-Up

% set filepath from main project folder (e.g. 'DATA/EEG/')
derivpath = '../../DATA/derivatives/';
sourcepath = '../../DATA/sourcedata/';
anpath = '../../ANALYSIS/';

% Subjects to include
subjects = {'sub-001','sub-002','sub-003','sub-004','sub-005'};

alltrials = 1:480;

%% prepare epoched data

% pupil data
load([derivpath,subjects{1},'/pupil/', subjects{1}, '_task-appav_pupil']);
pupil_fdbk = appav_pupil.fdbk_response;

% eeg data
load([derivpath,subjects{1},'/eeg/', subjects{1}, '_task-appav_epoched']); 
eeg_fdbk = allData(:,:,~appav_pupil.exclude_i);

%% H5 Data Structure
for i = 1:size(eeg_fdbk,3)

        trial = ['trial',num2str(i)];
        h5create('test.h5',['/test/',trial,'/eeg'],[1 1160]);
        h5write('test.h5',['/test/',trial,'/eeg'],eeg_fdbk(62,:,i));

end

for i = 1:size(pupil_fdbk,1)

        trial = ['trial',num2str(i)];
        h5create('test.h5',['/test/',trial,'/pupil'],[1 141]);
        h5write('test.h5',['/test/',trial,'/pupil'],pupil_fdbk(i,:));  
end

