import sys
sys.path.append('.')
import app.libs.ml_lib as lib
import scipy as sp
import numpy as np
import os
import pylab




if __name__ == '__main__':
    TRAINING_DATA_FILE = 'Gender_Detection/Train.txt'
    TEST_DATA_FILE = 'Gender_Detection/Test.txt'
    NR_FEATURES = 12
    D, L = lib.load_bniary_train_data(TRAINING_DATA_FILE, NR_FEATURES)
    lib.plot_hist_binary(D, L, NR_FEATURES, 'Female', 'Male')
    lib.plot_scatter(D, L, NR_FEATURES, 'Female', 'Male')


