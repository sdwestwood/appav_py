import os
import json
import h5py
import numpy as np
import statistics
import scipy.io as sci

# change directory to /DATA/ for reading in data
os.chdir('../../DATA')

# list of subjects to run
subjects = ['CM']
if len(subjects) != len(set(subjects)):
    print("Warning: duplicate subjects have been entered. Check your subjects list")

# Define epoching parameters
dur = 60  # Sliding window size (ms)
tbase1 = -100  # Baseline correction time (ms)
# run once with this baseline corrected and one with no correction
# (tbase1 = 0)

offset = [-100, 1000]
onsetT = 'tfdbk'

print("Beginning epoch loop per subject")

with open('events.json') as f:
    events = json.load(f)

print("Events file loaded")

for sj in subjects:
    print("Epoching subject: " + sj)

    #Read in filtered mat
    #Depending on the subject, either h5py or sci needs to be used, but both will not work for every subject
    #Ex: AM must use h5py, CM must use sci
    f = h5py.File('EEG/' + sj + '/' + sj + '_appav_filtered.mat', 'r')
    Y = f.get('Y')
    Y = np.array(Y)  # For converting to numpy array
    # Y = sci.loadmat('EEG/' + sj + '/' + sj + '_appav_filtered.mat')['Y']
    # Y = np.array(Y)

    print("Filtered EEG file loaded")
    # Changes rows and columns
    #TODO: check for the right shape before transpose, ex cm dosent need it
    Y = Y.transpose()

    allT = events[sj][onsetT]
    #Create blank arrays to append to
    allData = [[] for _ in range(len(Y))]

    for i in allT:
        baseline = []
        if tbase1 != 0:
            for column in Y:
                baseline.append(statistics.mean(column[i + tbase1:i]))
        else:
            baseline = 0

        counter = 0
        for column in Y:
            #Get the required slice, adjust for baseline, and add it to the right data list
            data = column[i + offset[0] - int(dur / 2) - 1:i + offset[-1] + int(dur / 2) - 1].tolist()
            data = [j - baseline[counter] for j in data]
            allData[counter] += data
            counter += 1

    allData = np.array(allData)
    #reshape the data to the required dimensions3
    allData = allData.reshape(len(allData), int(len(allData[0]) / len(allT)), len(allT), order='F').tolist()

    #Save subject data
    os.chdir('EEG/' + sj + '/')
    if not (tbase1):
        filename = sj + '_epoched.json'
    else:
        filename = sj + '_epoched_base.json'


    with open(filename, 'w') as f:
        json.dump(allData, f)

    print("Subject " + sj + " done!\n")
print("FINISHED EPOCHING ALL SUBJECTS")
