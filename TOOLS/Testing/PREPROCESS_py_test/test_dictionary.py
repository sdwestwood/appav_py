# function to convert matlab events structure to dictionary
import json
import scipy.io as sci
import os


#relative or absolute paths can be used
#TODO: adjust error message for column doesn't exist to be more specific

def dictionary(mat_filepath, json_filepath):
    mat_events = sci.loadmat(mat_filepath)[mat_filepath.split('/')[-1].split('.')[0]][0]
    with open(json_filepath) as f:
        json_events = json.load(f)

    mat_dict = {}
    mat_keys = [i for i in mat_events.dtype.fields]

    #build mat events dictionary
    for row in mat_events:
        mat_dict[row[0][0]] = {mat_keys[value]:row[value][0].tolist() for value in range(1, len(mat_keys))}

    difference = set(json_events.keys()).symmetric_difference(set(mat_dict.keys()))

    if len(difference):
        print("There are different/missing subjects in the event dictionaries. The differences are:")
        print(*difference, sep=", ")
        print("Continuing to compare values\n")

    customTestInLoop = ["nchid"]
    customTestOutLoop = []
    mat_keys = [value for value in mat_keys if value not in customTestOutLoop]



    subjects = list(json_events.keys())
    for subject in subjects:
        print(subject+":")
        subject_dict = json_events[subject]
        if not len(subject_dict):
            print(subject + " has no data saved")

        for column in mat_keys[1:]:
            if column in customTestInLoop:
                subject_dict[column] = [i+1 for i in subject_dict[column]]
            try:
                subject_dict[column]
            except:
                print("\t" + column + ": column does not exist in JSON version")
                continue
            try:
                if subject_dict[column] != mat_dict[subject][column]:
                    print("\t" +column+": different values found")
            except:
                print("\t"+column+": error occured. Usually column missing in mat version")


