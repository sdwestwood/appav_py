from inspect import getmembers, isfunction
import os
import numpy as np
import json
import scipy.io as sci

def resetPath():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

#Settings which may need to be passed to functions
def settings_setup():
    settings = {}
    #TODO: would be good to paths here so changes only need to happen once

    #TODO: this structure defeats the point of the settings idea. All settings should be base level dictionary, with specific functions having names of settings needed
    #general settings
    settings["path"] = "../TestData/"

    #getX
    #settings in order: cond_discr, offset, tbase, dur
    settings["getX"] = {}

    settings["getX"]["cond_discr"] = [1, 2]
    settings["getX"]["offset"] = np.arange(-100, 801, 10)
    settings["getX"]["dur"] = 60
    settings["getX"]["tbase"] = 100
    #settings["getX"] = [[1,2], np.arange(-100, 801, 10), 100, 60]

    #bernoull
    #TODO: no idea what v does
    settings["bernoull"] = {}
    settings["bernoull"]["v"] = ["something"]

    return settings

#participants = [f.name for f in os.scandir("TestData") if f.is_dir()]
import test_functions
#Get all the functions from test_functions
functions = getmembers(test_functions, isfunction)
#we only want functions that start with test_, and then remove test_ at the start. All functions from STA are here as well, so needs to be removed
functions = [i for i in functions if i[0][:5] == "test_"]
settings = settings_setup()
subjects = ["CM"]

for func in functions:
    #Remove "test_" from the function name
    print("\nTesting function: " + func[0][5:])
    for sj in subjects:
        try:
            #call the function with the participant and settings
            func[1](sj, settings[func[0][5:]], settings["path"])
        except:
            #TODO: better error messages
            print("\tError occured. continuing")

print("\nDone!")