import scipy.io as sci
import numpy as np
import os
from EEG_folders.Python.TOOLS.runLDA.Functions.STA_functions import *
import json
import statistics

#TODO: in suit, pass in participants and make this a function
def test_getX(sj, settings, path):
    print("\tTesting participant: " + sj)

    try:
        with open(path + 'events.json') as f:
            events = json.load(f)
    except:
        print("\t\tERROR: Problem with loading events, check path is right in settings")
        return

    try:
        path += sj + '/'
        with open(path + sj+"_epoched_base.json") as f:
            test_dat = json.load(f)
        test_dat = np.array(test_dat)
    except:
        print("\t\tERROR: Problem with loading subject epoch, check path is right in settings")
        return

    try:
        expected = sci.loadmat(path + 'test_X.mat')
    except:
        print("\t\tERROR: Problem loading test_X mat. Make sure this is stored in testdata, in the participant's folder")
        return

    try:
        expected_X = expected['X']
    except:
        print("\t\tERROR: X is not found in expected outcome. Make sure X values are stored as X us test_X mat file")
        return
    try:
        expected_Truth = expected['truth']
    except:
        print("\t\tERROR: truth is not found in expected outcome. Make sure truth values are stored as truth us test_X mat file")
        return

    try:
        cond_id = events[sj]['fdbk']
    except:
        print("\t\tERROR: trouble reading fdbk from subject's events. Make sure that fdbk exists in events")
        return

    #cond_id = [i for i in cond_id if i != 0]
    try:
        cond_id = np.array(cond_id)
    except:
        print("\t\tERROR: trouble converting fdbk list into numpy array")
        return

    try:
        cond_discr = settings["cond_discr"]
        offset = settings["offset"]
        tbase = settings["tbase"]
        dur = settings["dur"]
    except:
        print("\t\tERROR: trouble reading in setting for getX. make sure that cond_discr, offset, tbase, and dur exist in getX's settings")
        return

    try:
        test_getx = getX(test_dat, cond_id, cond_discr, offset[0] + tbase, dur)
    except:
        print("\t\tERROR: error occured in getX")
        return

    try:
        if test_getx['X'].tolist() != expected_X.tolist():
            print("\t\tX result does not match")

        if test_getx['truth'].tolist() != expected_Truth.tolist():
            print("\t\tTruth result does not match")
    except:
        print("\t\tERROR: trouble comparing results")

def test_bernoull(sj, settings, path):
    print("\tTesting participant: "+sj)
    v = settings["v"]

    with open(path + sj + '/' + sj + "_epoched_base.json") as f:
        X = json.load(f)
    X = np.array(X)
    y = X.transpose() * v[1:-2] + v[-1]
    result = bernoull(1, y)
    result_mean = bernoull(1, statistics.mean(y))
    result_median = bernoull(1, statistics.median(y))

    # These are the three ways its used
    Y_expected = "PATH/TO/Y"
    Y_mean_expected = "PATH/TO/Y"
    Y_median_expected = "PATH/TO/Y"

    if result == Y_expected:
        print("Y result does not match")
    if result_mean == Y_mean_expected:
        print("Y mean result dosent match")
    if result_median == Y_median_expected:
        print("Y median result dosent match")

# def test_rocarea(sj, settings):
#     p = "PATH/TO/p"
#     truth_aug = "PATH/TO/truth"
#     truthmean = "PATH/TO/truthmean"
#     ploo = "PATH/TO/ploo"
#     ploomedian = "PATH/TO/median"

