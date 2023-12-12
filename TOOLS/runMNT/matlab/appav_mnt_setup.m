%% Convert Pupil, EEG, and Behavioural Data to h5 format %%
clear all
close all

%% Set-Up

% Subjects to include
subjects = {'sub-001'};

for sidx=1:length(subjects)

    % set filepath from main project folder 
    filepath = 'DATA/derivatives/';
    sj = subjects{sidx};

    %% prepare epoched data
    
    % pupil data (trials x samples)
    load([filepath,sj,'/pupil/', subjects{sidx}, '_task-appav_pupil']);
    pupil_fdbk = appav_pupil.fdbk_response;
    
    % eeg data (channels x samples x trials)
    load([filepath,sj,'/eeg/', subjects{sidx}, '_task-appav_epoched']); 
    eeg_fdbk = allData(:,:,~appav_pupil.exclude_i);
    
    % beh data (truth label for correct response)
    load([filepath,sj,'/beh/', subjects{sidx}, '_task-appav_events.mat']); 
    acc = appav_events.acc;
    cond = int8(appav_events.isApp);
    
    %% H5 Data Structure
    fprintf('Saving eeg: %s\n',sj);
    for i = 1:size(eeg_fdbk,3)
    
            trial = ['trial',num2str(i)];
            h5create(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/eeg'],[1 1160]);
            h5write(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/eeg'],eeg_fdbk(62,:,i));
    end
    fprintf('Saving pupil: %s\n',sj);
    for i = 1:size(pupil_fdbk,1)
    
            trial = ['trial',num2str(i)];
            h5create(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/pupil'],[1 141]);
            h5write(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/pupil'],pupil_fdbk(i,:));  
    end
    fprintf('Saving accuracy: %s\n',sj);
    for i = 1:length(acc) %0=incorrect, 1=correct
    
            trial = ['trial',num2str(i)];
            h5create(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/acc'],1);
            h5write(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/acc'],acc(i));  
    end
    fprintf('Saving condition: %s\n',sj);
    for i = 1:length(cond) %0=punishment, 1=reward
    
            trial = ['trial',num2str(i)];
            h5create(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/cond'],1);
            h5write(['../data/',sj,'/',sj,'.h5'],['/',sj,'/',trial,'/cond'],cond(i));  
    end
end