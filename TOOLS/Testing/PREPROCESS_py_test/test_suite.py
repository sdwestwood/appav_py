import os
import test_dictionary
import json
import scipy.io as sci
#os.chdir("../../../DATA")
#TODO: appav_plot_eeg, choose oz, DATA/EEG vhdr file will tell electrode numbers
#Take oz epoch data, average across all trials, plot it
#Test get_events
print("Testing get_events")
mat_filepath = "../../../DATA/all_appav_events.mat"
json_filepath = "../../../DATA/events.json"
test_dictionary.dictionary(mat_filepath, json_filepath)

#Testing epoch
print("Testing epoch")
os.chdir("EEG/AM")
mat_filepath = "AM_epoched_base.mat"
json_filepath = "AM_epoched_base.json"

mat_epoch = sci.loadmat(mat_filepath)["allData"].tolist()
with open(json_filepath) as f:
    json_epoch = json.load(f)

flag = True
for i in range(len(mat_epoch)):
    for j in range(len(mat_epoch[i])):
        for k in range(len(mat_epoch[i][j])):
            if "%.5f" % mat_epoch[i][j][k] != "%.5f" % json_epoch[i][j][k]:
                print(str(i) + " " + str(j) + " is wrong")
                flag = False
if flag:
    print("epochs work")