#import unittest

#import matplotlib as plot
import numpy as np
import scipy.io as sci
# import STA_functions.bernoull as bernoull
import os
from EEG_folders.Python.TOOLS.runLDA.Functions.STA_functions import getX

test_dat = sci.loadmat('CM_epoched_base.mat')
test_output = sci.loadmat('test_X.mat')
test_X = test_output['X']
test_truth = test_output['truth']
test_mat = test_dat['allData']
cond_discr = [1, 2]
cond_id = np.tile([1, 2, 2, 1], 40)
offset = np.arange(-100, 801, 10)
dur = 60
tbase = 100


test_getx = getX(test_mat, cond_id, cond_discr, offset[0] + tbase, dur)

new_X = test_getx['X']
new_truth = test_getx['truth']


print(new_X == test_X)





