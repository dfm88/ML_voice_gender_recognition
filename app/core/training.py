import sys
sys.path.append('.')
import app.libs.ml_lib as lib
import scipy as sp
import numpy as np
import os
import pylab

PROVAAAAAAAAA = 'voice'

if PROVAAAAAAAAA == 'voice':
    TRAINING_DATA_FILE = 'Gender_Detection/Train.txt'
    TEST_DATA_FILE = 'Gender_Detection/Test.txt'
    NR_FEATURES = 12
elif PROVAAAAAAAAA == 'pulsar':
    TRAINING_DATA_FILE = 'Pulsar_Detection/Train.txt'
    TEST_DATA_FILE = 'Pulsar_Detection/Test.txt'
    NR_FEATURES = 8



if __name__ == '__main__':

    D, L = lib.load_bniary_train_data(TRAINING_DATA_FILE, NR_FEATURES)
    # lib.plot_hist_binary(D, L, NR_FEATURES, 'Female', 'Male') # non Gaussian 10-9-7-0-3-2
    # lib.plot_scatter(D, L, NR_FEATURES, 'Female', 'Male')
    # lib.plot_pearson_heatmap(D, L)


