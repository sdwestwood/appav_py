""" 
Main analysis script for running LDA
"""

import numpy as np

from methods.py import logist, logistpca, ldareg
from single_trial_analysis.py import single_trial_analysis
from STA_functions.py import get_X, geninv, rocarea, bernoull

