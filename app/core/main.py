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


def plotting(D, L):
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)

    ## PLOTTING
    lib.plot_hist_binary(D_norm, L, NR_FEATURES, 'Female', 'Male') # non Gaussian 10-9-7-0-3-2
    lib.plot_scatter(D_norm, L, NR_FEATURES, 'Female', 'Male')
    lib.plot_pearson_heatmap(D_norm, L)
    
    ## GAUSSIANIZATION
    lib.plot_hist_binary(D_norm_gau, L, NR_FEATURES, 'Female', 'Male', 'gaussianized')

def gaussian(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    GAUSSIAN TRAINING
    """
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)

    for i in range(4): 
        D_pca = lib.PCA(D_norm, m=D_norm.shape[0]-i)
        D_pca_gau = lib.PCA(D_norm_gau, m=D_norm_gau.shape[0]-i)

        ###   - - - - -      GAUSSIAN NORMAL  - - - - -    ####
        print(f'\nGAUSSIAN FULL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf = lib.K_fold(D_pca, L, GaussianClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
            print("min DCF MVG Full-Cov with prior=%.1f:  %.3f" %(prior_cl_T, min_dcf))
            print("min DCF MVG `gaussianized` Full-Cov with prior=%.1f:  %.3f\n" %(prior_cl_T, min_dcf_gau))

        ###   - - - - -      GAUSSIAN BAYES  - - - - -    ####
        print(f'\nGAUSSIAN DIAGONAL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf = lib.K_fold(D_pca, L, GaussianBayesianClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianBayesianClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
            print("min DCF MVG Diagonal-Cov with prior=%.1f:  %.3f" %(prior_cl_T, min_dcf))
            print("min DCF MVG `gaussianized` Diagonal-Cov with prior=%.1f:  %.3f\n" %(prior_cl_T, min_dcf_gau))


        ###   - - - - -      GAUSSIAN TIED  - - - - -    ####
        print(f'\nGAUSSIAN TIED COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf = lib.K_fold(D_pca, L, GaussianTiedClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
            min_dcf_gau = lib.K_fold(D_pca_gau, L, GaussianTiedClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn)
        
            print("min DCF MVG Tied-Cov with prior=%.1f:  %.3f" %(prior_cl_T, min_dcf))
            print("min DCF MVG `gaussianized` Tied-Cov with prior=%.1f:  %.3f\n" %(prior_cl_T, min_dcf_gau))


def linear_logistic_regression(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    LOGISTIC LINEAR REGRESSION TRAINING
    """
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)
    # ###   - - - - -      LOGISTIC REGRESSION  - - - - -    ####
    print(f'\nLOGISTIC REGRESSION WITH K FOLD ({nr_kfold_split} folds) ')
    _lambdas=np.logspace(-5, 5, num=30)

    def lambda_tuning(prior_cl_T, _lambdas, regularized=True, pi_T=0.5):
        """
        returns 2 list 
            one with lambda estimation for Raw features
            one with lambda estimation for Gaussianized features
    
        """
        min_DCF_z_regul_list = []
        min_DCF_gau_regul_list = []
        for _l in _lambdas:
            min_dcfF_z_not_regul = lib.K_fold(D_norm, L, LogisticRegressionClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, _lambda=_l, regularized=regularized, pi_T=pi_T)
            min_DCF_z_regul_list.append(min_dcfF_z_not_regul)
            min_dcf_gau_not_regul = lib.K_fold(D_norm_gau, L, LogisticRegressionClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, _lambda=_l, regularized=regularized, pi_T=pi_T)
            min_DCF_gau_regul_list.append(min_dcf_gau_not_regul)
            print(f"min DCF LOGISTIC REGRESSION 'z' REG by {pi_T} with prior={prior_cl_T} and lambda={_l}:  {min_dcfF_z_not_regul:.3f}")
            print(f"min DCF LOGISTIC REGRESSION 'Gaussuanized' REG by {pi_T} with prior={prior_cl_T} and lambda={_l}:  {min_dcf_gau_not_regul:.3f}")
        return (min_DCF_z_regul_list, min_DCF_gau_regul_list)


    ### Estimating for different values of lambda
    # tot_z_reg = []
    # tot_gau_reg=[]
    # pi_T=0.5
    # regularized=True
    # for prior in application_priors:
    #     print(f'\n -- ------  APPLICATION PRIOR {prior}')
    #     min_DCF_z_regul_list, min_DCF_gau_regul_list = lambda_tuning(prior, _lambdas, regularized=regularized, pi_T=pi_T)
    #     tot_z_reg = tot_z_reg + min_DCF_z_regul_list
    #     tot_gau_reg = tot_gau_reg + min_DCF_gau_regul_list
    # print('\n\nmin DCF for Z')
    # print(min(tot_z_reg))
    # print('\n\nmin DCF for gau')
    # print(min(tot_gau_reg))
    # lib.plotDCF(_lambdas, tot_z_reg, 'lambda', 'LR_z_regularized', regularized=regularized, pi_T=pi_T)
    # lib.plotDCF(_lambdas, tot_gau_reg, 'lambda', 'LR_gau_regularized', regularized=regularized, pi_T=pi_T)


    # using lambda = 0 from previous result, we'll try, for each application prior
    # different empirical priors
    _lambda = 0
    for prior_cl_T in application_priors:
        for pi_T in application_priors:
            print(f'\n----- Prior {prior_cl_T} --- pi_T {pi_T}')
            min_dcf_z = lib.K_fold(D_norm, L, LogisticRegressionClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, _lambda=_lambda, regularized=True, pi_T=pi_T)
            min_dcf_gau = lib.K_fold(D_norm_gau, L, LogisticRegressionClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, _lambda=_lambda, regularized=True, pi_T=pi_T)
            print(f"min DCFLOGISTIC REGRESSION 'z' and lambda={_lambda}:  {min_dcf_z:.3f}")
            print(f"min DCFLOGISTIC REGRESSION 'Gaussian' and lambda={_lambda}:  {min_dcf_gau:.3f}")

def svm_linear(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    SVM LINEAR TRAINING
    """
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)
    # ###   - - - - -      SVM LINEAR REGRESSION  - - - - -    ####
    print(f'\SVM LINEAR WITH K FOLD ({nr_kfold_split} folds) ')
    C_list = np.logspace(-3, 1, num=30)

    def C_tuning(prior_cl_T):
        """
        returns 2 list 
            one with C estimation for Raw features
            one with C estimation for Gaussianized features
    
        """
        min_DCF_z_list = []
        min_DCF_gau_list = []
        for c in C_list:
            min_dcfF_z = lib.K_fold(D_norm, L, SVMLinearClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, C=c)
            min_DCF_z_list.append(min_dcfF_z)
            min_dcf_gau = lib.K_fold(D_norm, L, SVMLinearClassifier, k=nr_kfold_split, prior_cl_T=prior_cl_T, cfp=cfp, cfn=cfn, C=c)
            min_DCF_gau_list.append(min_dcf_gau)
            print(f"min DCF SVM Linear 'z' with C:{c} and prior {prior_cl_T}:  {min_dcfF_z:.3f}")
            print(f"min DCF SVM Linear 'Gaussuanized' with C:{c} and prior {prior_cl_T}:  {min_dcf_gau:.3f}")
        return (min_DCF_z_list, min_DCF_gau_list)

    ### Estimating for different values of C
    tot_z_reg = []
    tot_gau_reg=[]
    # pi_T=0.5
    # regularized=True
    for prior in application_priors:
        print(f'\n -- ------  APPLICATION PRIOR {prior}')
        min_DCF_z_regul_list, min_DCF_gau_regul_list = C_tuning(prior_cl_T=prior)
        tot_z_reg = tot_z_reg + min_DCF_z_regul_list
        tot_gau_reg = tot_gau_reg + min_DCF_gau_regul_list
    print('\n\nmin DCF for Z')
    print(min(tot_z_reg))
    print('\n\nmin DCF for gau')
    print(min(tot_gau_reg))
    lib.plotDCF(C_list, tot_z_reg, 'C', 'SVM_linear_z')
    lib.plotDCF(C_list, tot_gau_reg, 'C', 'SVM_linear_gau')


if __name__ == '__main__':
    import app.libs.ml_lib as lib
    from app.core.classifiers import (
        GaussianClassifier, 
        GaussianBayesianClassifier, 
        GaussianTiedClassifier,
        LogisticRegressionClassifier,
        SVMLinearClassifier
    )

    D, L = lib.load_binary_data(TRAINING_DATA_FILE, NR_FEATURES)
    # D, L = lib.load_iris_binary()
    application_priors = [0.5, 0.9, 0.1]
    D_norm = lib.z_normalization(D)
    D_norm_gau = lib.gaussianization(D_norm, D_norm)

    nr_kfold_split = 4
    cfp = 1
    cfn = 1


    # plotting(D, L)

    # gaussian(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # linear_logistic_regression(D, L, application_priors, nr_kfold_split, cfp, cfn)

    svm_linear(D, L, application_priors, nr_kfold_split, cfp, cfn)










