import sys

sys.path.append('.')
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
    import app.libs.ml_lib as lib
    from app.core.classifiers import GaussianClassifier, GaussianBayesianClassifier, GaussianTiedClassifier

    D, L = lib.load_binary_train_data(TRAINING_DATA_FILE, NR_FEATURES)
    # D, L = lib.load_iris_binary()

    ### PLOTTING
    # lib.plot_hist_binary(D_norm, L, NR_FEATURES, 'Female', 'Male') # non Gaussian 10-9-7-0-3-2
    # lib.plot_scatter(D_norm, L, NR_FEATURES, 'Female', 'Male')
    # lib.plot_pearson_heatmap(D_norm, L)
    
    ### GAUSSIANIZATION
    # lib.plot_hist_binary(D_norm_gau, L, NR_FEATURES, 'Female', 'Male', 'gaussianized')


    different_priors_T = [0.5, 0.9, 0.1]
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)
    nr_kfold_split = 4
    cfp = 1
    cfn = 1

    for i in range(4): 
        D_pca = lib.PCA(D_norm, m=D_norm.shape[0]-i)
        D_pca_gau = lib.PCA(D_norm_gau, m=D_norm_gau.shape[0]-i)

        ###   - - - - -      GAUSSIAN NORMAL  - - - - -    ####
        print(f'\nGAUSSIAN FULL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_T in different_priors_T:
            min_dcf = lib.K_fold(D_pca, L, GaussianClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
            print("min DCF MVG Full-Cov with prior=%.1f:  %.3f" %(prior_T, min_dcf))
            print("min DCF MVG `gaussianized` Full-Cov with prior=%.1f:  %.3f\n" %(prior_T, min_dcf_gau))

        ###   - - - - -      GAUSSIAN BAYES  - - - - -    ####
        print(f'\nGAUSSIAN DIAGONAL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_T in different_priors_T:
            min_dcf = lib.K_fold(D_pca, L, GaussianBayesianClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianBayesianClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
            print("min DCF MVG Diagonal-Cov with prior=%.1f:  %.3f" %(prior_T, min_dcf))
            print("min DCF MVG `gaussianized` Diagonal-Cov with prior=%.1f:  %.3f\n" %(prior_T, min_dcf_gau))


        ###   - - - - -      GAUSSIAN TIED  - - - - -    ####
        print(f'\nGAUSSIAN TIED COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_T in different_priors_T:
            min_dcf = lib.K_fold(D_pca, L, GaussianTiedClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianTiedClassifier, k=nr_kfold_split, prior_cl_T=prior_T, cfp=cfp, cfn=cfn)
        
            print("min DCF MVG Tied-Cov with prior=%.1f:  %.3f" %(prior_T, min_dcf))
            print("min DCF MVG `gaussianized` Tied-Cov with prior=%.1f:  %.3f\n" %(prior_T, min_dcf_gau))







