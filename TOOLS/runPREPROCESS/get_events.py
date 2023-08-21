import json
import numpy as np
from functions import read_brain_recorder_vmrk


# list of subjects to run
subjects = []
if len(subjects) != len(set(subjects)):
    print("Warning: duplicate subjects have been entered. Check your subjects list")

#TODO: the following wont work with participants that dosent have 6 blocks, such as CM
block_range = range(1, 7)  # list of 1:nblocks


# Event codes sent to parallel port
stimcodes = [25, 40, 1, 2, 48, 49, 50]  # relevant event codes (see below)

# add list of port code events here for reference e.g.:
# 24 - trial start (currently not included)
# 25 - no choice
# 32 - inter trial fixation 1
# 33 - inter trial fixation 2
# 40 - trial choice present
# 1 = left symbol chosen
# 2 = right symbol chosen
# 48 - positive fdbk
# 49 - neutral fdbk
# 50 - negative fdbk


########## Create Events Dictionary ##########

events = {}

# set field names in the correct order in which they are organised in tmplog
fields = ['hisym','choice','acc','fdbk','tstim','tchoice','tfdbk','rt','cond','block','newblock','nchid']
col_idx = [6, 3, 0, 2]

filepath = "../../DATA/"
# Begin events loop
for sj in subjects:

    # read in vmrk file
    print(" Extracting events from subject: " + sj)
    tmpevents = read_brain_recorder_vmrk(filepath, ('EEG/' + sj + '/' + sj + '_appav.vmrk'), stimcodes)

    # extract eeg vars from tmpevents
    tstim = [i[1] for i in tmpevents if i[0] == 40]
    tmp_idx = range(len(tmpevents))
    tchoice = [tmpevents[i + 1][1] for i in tmp_idx if tmpevents[i][0] == 40]
    tfdbk = [i[1] for i in tmpevents if i[0] in [48, 49, 50]]
    rt = [tch - tst for tch, tst in zip(tchoice, tstim)]
    tmpeeg = [tstim, tchoice, tfdbk, rt]
    del tmp_idx, tmpevents

    # initialise field names and empty tmplog list
    tmplog = []
    for bidx in block_range:
        # load in block logfile
        with open(filepath+'Pupil/' + sj + '/' + sj + '_' + str(bidx) + ".txt") as f:
            block = f.read().split("\n")
            block.pop()
            tmplog+=block

    # extract desired variables and append to tmplog
    tmplog = [i.split("\t") for i in tmplog]
    tmplog = [list(map(int, i)) for i in tmplog]
    tmplog = np.array(tmplog)
    tmplog = tmplog.transpose()
    tmplog = [tmplog[i] for i in col_idx]  # retain necessary columns and rearrange to match matlab

    del bidx, f

    # append eeg vars to tmplog as np.arrays
    [tmplog.append(i) for i in tmpeeg]

    # index no choice trials
    tmp_nchid = np.where(tmplog[3] == -2)[0]

    # add condition and block fields
    tmplog.append(np.repeat([1, 2] * 3, 80))  # cond
    tmplog.append(np.repeat(block_range, 80))  # block
    tmplog.append(np.tile(np.array([1]+[0]*79),6)) # newblock
    

    # delete items in tmplog at nchid indices
    tmplog = [np.delete(i, tmp_nchid) for i in tmplog]
    tmplog[6] = np.array(tfdbk)  # reinstate tfdbk
    tmplog.append(tmp_nchid)  # add nchid field

    # create sj dictionary as entry in events dictionary
    events[sj] = {i: np.array(j) for i, j in zip(fields, tmplog)}

    del tmplog, tmpeeg, tstim, tchoice, tfdbk, rt, tmp_nchid
del sj, block_range, stimcodes, subjects, fields, col_idx

for sj in events:
    for i in events[sj].keys():
        events[sj][i] = events[sj][i].tolist()
    del i
del sj

with open(filepath+'events.json', 'w') as f:
    # default_flow_style=False makes yml file more readable
    # 'True' saves a little memory
    json.dump(events, f)
del f, filepath

# # to load events
# with open('events.yml') as f:
#     events = yaml.load(f, Loader=yaml.FullLoader)