import sys
sys.path.append('.')
import scipy as sp
import numpy as np
import os
import pylab

PROVAAAAAAAAA = 'pulsar'

if PROVAAAAAAAAA == 'voice':
    TRAINING_DATA_FILE = 'Gender_Detection/Train.txt'
    TEST_DATA_FILE = 'Gender_Detection/Test.txt'
    NR_FEATURES = 12
elif PROVAAAAAAAAA == 'pulsar':
    TRAINING_DATA_FILE = 'Pulsar_Detection/Train.txt'
    TEST_DATA_FILE = 'Pulsar_Detection/Test.txt'
    NR_FEATURES = 8



if __name__ == '__main__':
    import app.libs.ml_lib as lib
    from app.core.classifiers import GaussianClassifier

    # D, L = lib.load_binary_train_data(TRAINING_DATA_FILE, NR_FEATURES)
    D, L = lib.load_iris_binary_reduced()
    D, L = lib.load_iris_binary()
    import ipdb; ipdb.set_trace()
    (DTR, LTR), (DTE, LTE) = lib.split_db_2to1(D, L)
    gc = GaussianClassifier(DTR, LTR)
    gc.train(DTE, [0.5, 0.5])
    gc.posteriors
    import ipdb; ipdb.set_trace()
    gc.classify(LTE)
    import ipdb; ipdb.set_trace()



    
    D_norm = lib.z_normalization(D)

    D = lib.PCA(D_norm, m=D.shape[1])

    lib.K_fold(D, L, GaussianClassifier, k=3)

    ### PLOTTING
    # lib.plot_hist_binary(D_norm, L, NR_FEATURES, 'Female', 'Male') # non Gaussian 10-9-7-0-3-2
    # lib.plot_scatter(D_norm, L, NR_FEATURES, 'Female', 'Male')
    # lib.plot_pearson_heatmap(D_norm, L)
    
    ### GAUSSANIZATION
    D_norm_gau = lib.gaussanization(D_norm, D_norm)
    # lib.plot_hist_binary(D_norm_gau, L, NR_FEATURES, 'Female', 'Male', 'gaussianized')





