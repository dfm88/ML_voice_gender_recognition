from functools import partial
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


def plotting(D, L):
    D_norm, _ = lib.z_normalization(D)
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
    for i in range(4): 
        ###   - - - - -      GAUSSIAN NORMAL  - - - - -    ####
        print(f'\nGAUSSIAN FULL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            min_dcf_gau, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianClassifier, 
                z_norm=True, 
                gaus=True, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            print(f"min DCF MVG Full-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f" %(prior_cl_T, min_dcf))
            print(f"min DCF MVG `gaussianized` Full-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f\n" %(prior_cl_T, min_dcf_gau))

        ###   - - - - -      GAUSSIAN BAYES  - - - - -    ####
        print(f'\nGAUSSIAN DIAGONAL COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianBayesianClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            min_dcf_gau, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianBayesianClassifier, 
                z_norm=True, 
                gaus=True, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            print(f"min DCF MVG Diagonal-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f" %(prior_cl_T, min_dcf))
            print(f"min DCF MVG `gaussianized` Diagonal-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f\n" %(prior_cl_T, min_dcf_gau))


        ###   - - - - -      GAUSSIAN TIED  - - - - -    ####
        print(f'\nGAUSSIAN TIED COV WITH K FOLD ({nr_kfold_split} folds) ')
        for prior_cl_T in application_priors:
            min_dcf, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianTiedClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            min_dcf_gau, _, _ = lib.K_fold(
                D, 
                L, 
                GaussianTiedClassifier, 
                z_norm=True, 
                gaus=True, 
                pca_m=D.shape[0]-i, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn
            )
            print(f"min DCF MVG Tied-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f" %(prior_cl_T, min_dcf))
            print(f"min DCF MVG `gaussianized` Tied-Cov with prior=%.1f, PCA={D.shape[0]-i}/{D.shape[0]}:  %.3f\n" %(prior_cl_T, min_dcf_gau))


def linear_logistic_regression(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    LOGISTIC LINEAR REGRESSION TRAINING
    """
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
            min_dcfF_z_not_regul, _, _ = lib.K_fold(
                D, 
                L, 
                LogisticRegressionClassifier,
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                _lambda=_l, 
                regularized=regularized, 
                pi_T=pi_T,
            )
            min_DCF_z_regul_list.append(min_dcfF_z_not_regul)
            min_dcf_gau_not_regul, _, _ = lib.K_fold(
                D, 
                L, 
                LogisticRegressionClassifier,
                z_norm=True, 
                gaus=True, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                _lambda=_l, 
                regularized=regularized, 
                pi_T=pi_T,
            )
            min_DCF_gau_regul_list.append(min_dcf_gau_not_regul)
            print(f"min DCF LOGISTIC REGRESSION 'z' REG by {pi_T} with prior={prior_cl_T} and lambda={_l}:  {min_dcfF_z_not_regul:.3f}")
            print(f"min DCF LOGISTIC REGRESSION 'Gaussianized' REG by {pi_T} with prior={prior_cl_T} and lambda={_l}:  {min_dcf_gau_not_regul:.3f}")
        return (min_DCF_z_regul_list, min_DCF_gau_regul_list)


    ### Estimating for different values of lambda
    tot_z_reg = []
    tot_gau_reg=[]
    pi_T=0.5
    regularized=True
    for prior in application_priors:
        print(f'\n -- ------  APPLICATION PRIOR {prior}')
        min_DCF_z_regul_list, min_DCF_gau_regul_list = lambda_tuning(prior, _lambdas, regularized=regularized, pi_T=pi_T)
        tot_z_reg = tot_z_reg + min_DCF_z_regul_list
        tot_gau_reg = tot_gau_reg + min_DCF_gau_regul_list
    print('\n\nmin DCF for Z')
    print(min(tot_z_reg))
    print('\n\nmin DCF for gau')
    print(min(tot_gau_reg))
    lib.plot_dcf(_lambdas, tot_z_reg, 'lambda', 'LR_z_regularized', regularized=regularized, pi_T=pi_T)
    lib.plot_dcf(_lambdas, tot_gau_reg, 'lambda', 'LR_gau_regularized', regularized=regularized, pi_T=pi_T)


    # using lambda = 0 from previous result, we'll try, for each application prior
    # different empirical priors
    _lambda = 0
    ## REGULARIZED
    for prior_cl_T in application_priors:
        for pi_T in application_priors:
            print(f'\n----- Prior {prior_cl_T} --- pi_T {pi_T}')
            min_dcf_z, _, _ = lib.K_fold(
                D, 
                L, 
                LogisticRegressionClassifier,
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                _lambda=_lambda, 
                regularized=True, 
                pi_T=pi_T,
            )
            min_dcf_gau, _, _ = lib.K_fold(
                D, 
                L, 
                LogisticRegressionClassifier,
                z_norm=True, 
                gaus=True, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                _lambda=_lambda, 
                regularized=True, 
                pi_T=pi_T,
            )
            print(f"min DCFLOGISTIC REGRESSION 'z' and lambda={_lambda}:  {min_dcf_z:.3f}")
            print(f"min DCFLOGISTIC REGRESSION 'Gaussian' and lambda={_lambda}:  {min_dcf_gau:.3f}")
 
    ## NOT REGULARIZED
    for prior_cl_T in application_priors:
        print(f'\n----- Prior {prior_cl_T} --- not regularized')
        min_dcf_z, _, _ = lib.K_fold(
            D, 
            L, 
            LogisticRegressionClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None, 
            k=nr_kfold_split, 
            prior_cl_T=prior_cl_T, 
            cfp=cfp, 
            cfn=cfn, 
            _lambda=_lambda, 
            regularized=False,
        )
        min_dcf_gau, _, _ = lib.K_fold(
            D, 
            L, 
            LogisticRegressionClassifier,
            z_norm=True, 
            gaus=True, 
            pca_m=None, 
            k=nr_kfold_split, 
            prior_cl_T=prior_cl_T, 
            cfp=cfp, 
            cfn=cfn, 
            _lambda=_lambda, 
            regularized=False,
        )
        print(f"min DCFLOGISTIC REGRESSION 'z' and lambda={_lambda}:  {min_dcf_z:.3f}")
        print(f"min DCFLOGISTIC REGRESSION 'Gaussian' and lambda={_lambda}:  {min_dcf_gau:.3f}")

def svm_linear(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    SVM LINEAR TRAINING
    """
    # ###   - - - - -      SVM LINEAR  - - - - -    ####
    print(f'\SVM LINEAR WITH K FOLD ({nr_kfold_split} folds) ')
    C_list = np.logspace(-3, 1, num=20) 

    def C_tuning(prior_cl_T):
        """
        returns 1 list 
            with C estimation for Raw features
    
        """
        min_DCF_z_list = []
        for c in C_list:
            min_dcfF_z, _, _ = lib.K_fold(
                D, 
                L, 
                SVMLinearClassifier, 
                z_norm=True,
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                C=c
            )
            min_DCF_z_list.append(min_dcfF_z)
            print(f"min DCF SVM Linear 'z' with C:{c} and prior {prior_cl_T}:  {min_dcfF_z}")
        return min_DCF_z_list

    # ## Estimating for different values of C
    tot_z_reg = []
    for prior in application_priors:
        print(f'\n -- ------  APPLICATION PRIOR {prior}')
        min_DCF_z_regul_list = C_tuning(prior_cl_T=prior)
        tot_z_reg = tot_z_reg + min_DCF_z_regul_list
    print('\n\nmin DCF for Z')
    print(min(tot_z_reg))
    lib.plot_dcf(C_list, tot_z_reg, 'C', 'SVM_linear_z')

    # using C = 1 from previous result, we'll try, for each application prior
    # different empirical priors
    C = 1
    ## REGULARIZED
    for prior_cl_T in application_priors:
        for pi_T in application_priors:
            print(f'\n----- Prior {prior_cl_T} --- pi_T {pi_T}')
            min_dcf_z, _, _ = lib.K_fold(
                D, 
                L, 
                SVMLinearClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                C=C, 
                rebalanced=True, 
                pi_T=pi_T
            )
            print(f"min DCF LINEAR CLASSIFIER 'z' and C={C}:  {min_dcf_z:.3f}")

    ## NOT REGULARIZED
    for prior_cl_T in application_priors:
        print(f'\n----- Prior {prior_cl_T} --- not regularized')
        min_dcf_z, _, _ = lib.K_fold(
            D, 
            L, 
            SVMLinearClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None, 
            k=nr_kfold_split, 
            prior_cl_T=prior_cl_T, 
            cfp=cfp, 
            cfn=cfn, 
            C=C, 
            rebalanced=False
        )
        print(f"min DCF LINEAR CLASSIFIER 'z' and C={C}:  {min_dcf_z:.3f}")


def svm_kernel_rbf(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    SVM KERNEL RBF TRAINING
    """
    # ###   - - - - -      SVM KERNEL RBF  - - - - -    ####
    print(f'\SVM KERNEL RBF WITH K FOLD ({nr_kfold_split} folds) ')
    C_list = np.logspace(-3, 1, num=20)
    gamma_list = [10**(-2), 10**(-1), 10**(0)]
    PRIOR = 0.5 # since we have 2 hyperparameters (C and gamma) we'll only consider our main application prior

    def C_gamma_tuning(prior_cl_T):
        """
        returns 1 list 
            with C estimation for Raw features for different values of gamma
    
        """
        min_DCF_z_list = []
        for gamma in gamma_list:
            for c in C_list:
                min_dcfF_z, _, _ = lib.K_fold(
                    D, 
                    L, 
                    SVMKernelRBFClassifier, 
                    z_norm=True, 
                    gaus=False, 
                    pca_m=None, 
                    k=nr_kfold_split, 
                    prior_cl_T=prior_cl_T, 
                    cfp=cfp, 
                    cfn=cfn, 
                    C=c, 
                    gamma=gamma
                )
                min_DCF_z_list.append(min_dcfF_z)
                print(f"min DCF SVM  KERNEL RBF 'z' with C:{c} and gamma:{gamma} and prior {prior_cl_T}:  {min_dcfF_z}")
        return min_DCF_z_list

    # ## Estimating for different combinations of C-gamma
    print(f'\n -- ------  APPLICATION PRIOR {PRIOR}')
    min_DCF_z_regul_list = C_gamma_tuning(prior_cl_T=PRIOR)
    print('\n\nmin DCF for Z')
    print(min(min_DCF_z_regul_list))
    lib.plot_dcf_kernelSVM(
        x=C_list, 
        y=min_DCF_z_regul_list, 
        xlabel=f'C', 
        model_name=f'SVM_RBF_z_prior_{PRIOR}',
        hyperpar_name='\u03B3',
        hyperpar_list=gamma_list
    )

    # using C = ? and gamma = ? from previous result, we'll try, for each application prior
    # different empirical priors
    C = 1
    gamma = '?'
    # REGULARIZED
    for prior_cl_T in application_priors:
        for pi_T in application_priors:
            print(f'\n----- Prior {prior_cl_T} --- pi_T {pi_T}')
            min_dcf_z, _, _ = lib.K_fold(
                D, 
                L, 
                SVMKernelRBFClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                C=C, 
                rebalanced=True, 
                pi_T=pi_T,
            )
            print(f"min DCF  KERNEL RBF CLASSIFIER 'z' and C={C}:  {min_dcf_z:.3f}")

    # NOT REGULARIZED
    for prior_cl_T in application_priors:
        print(f'\n----- Prior {prior_cl_T} --- not regularized')
        min_dcf_z, _, _ = lib.K_fold(
            D, 
            L, 
            SVMKernelRBFClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None, 
            k=nr_kfold_split, 
            prior_cl_T=prior_cl_T, 
            cfp=cfp, 
            cfn=cfn, 
            C=C, 
            rebalanced=False
        )
        print(f"min DCF  KERNEL RBF CLASSIFIER 'z' and C={C}:  {min_dcf_z:.3f}")


def svm_kernel_polynomial(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    SVM KERNEL POLYNOMIAL DEGREE 2 TRAINING
    """
    # ###   - - - - -      SVM KERNEL POLYNOMIAL DEGREE 2  - - - - -    ####
    print(f'\SVM KERNEL POLYNOMIAL DEGREE 2 WITH K FOLD ({nr_kfold_split} folds) ')
    C_list = np.logspace(-3, 1, num=20)
    _c_list = [0, 1, 10]
    PRIOR = 0.5 # since we have 2 hyperparameters (C and _c) we'll only consider our main application prior

    def C_c_tuning(prior_cl_T):
        """
        returns 1 list 
            with C estimation for Raw features for different values of c
    
        """
        min_DCF_z_list = []
        for _c in _c_list:  
            for c in C_list:
                min_dcfF_z, _, _ = lib.K_fold(
                    D, 
                    L, 
                    SVMKernelPolynomialClassifier, 
                    z_norm=True, 
                    gaus=False, 
                    pca_m=None, 
                    k=nr_kfold_split, 
                    prior_cl_T=prior_cl_T, 
                    cfp=cfp, 
                    cfn=cfn, 
                    C=c, 
                    _c=_c,
                )
                min_DCF_z_list.append(min_dcfF_z)
                print(f"min DCF SVM KERNEL POLYNOMIAL DEGREE 2 'z' with C:{c} and _c:{_c} and prior {prior_cl_T}:  {min_dcfF_z}")
        return min_DCF_z_list

    ## Estimating for different combinations of C-c
    print(f'\n -- ------  APPLICATION PRIOR {PRIOR}')
    min_DCF_z_regul_list = C_c_tuning(prior_cl_T=PRIOR)
    print('\n\nmin DCF for Z')
    print(min(min_DCF_z_regul_list))
    lib.plot_dcf_kernelSVM(
        x=C_list, 
        y=min_DCF_z_regul_list, 
        xlabel='C', 
        model_name=f'SVM_POLYNOMIAL_z_prior_{PRIOR}', 
        hyperpar_name='c',
        hyperpar_list=_c_list
    )

    # using C = 1 and _c = 10 from previous result, we'll try, for each application prior
    # different empirical priors
    C = 1
    _c = 10
    ## REGULARIZED
    for prior_cl_T in application_priors:
        for pi_T in application_priors:
            print(f'\n----- Prior {prior_cl_T} --- pi_T {pi_T}')
            min_dcf_z, _, _ = lib.K_fold(
                D, 
                L, 
                SVMKernelPolynomialClassifier, 
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_cl_T, 
                cfp=cfp, 
                cfn=cfn, 
                C=C, 
                _c=_c, 
                rebalanced=True, 
                pi_T=pi_T
            )
            print(f"min DCF KERNEL POLYNOMIAL DEGREE 2 CLASSIFIER 'z' and C={C} and _c={_c}:  {min_dcf_z:.3f}")

    ## NOT REGULARIZED
    for prior_cl_T in application_priors:
        print(f'\n----- Prior {prior_cl_T} --- not regularized')
        min_dcf_z, _, _ = lib.K_fold(
            D, 
            L, 
            SVMKernelPolynomialClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None, 
            k=nr_kfold_split, 
            prior_cl_T=prior_cl_T, 
            cfp=cfp, 
            cfn=cfn, 
            C=C, 
            _c=_c, 
            rebalanced=False
        )
        print(f"min DCF KERNEL POLYNOMIAL DEGREE 2 CLASSIFIER 'z' and C={C} and _c={_c}:  {min_dcf_z:.3f}")


def gmm(D, L, application_priors:list, nr_kfold_split, cfp, cfn):
    """
    GMM TRAINING
    """
    # ###   - - - - -      GMM  - - - - -    ####
    print(f'\GMM WITH K FOLD ({nr_kfold_split} folds) ')


    ###   - - - - -  GMM GAUSSIAN NORMAL  - - - - -    ####
    print(f'\nGMM GAUSSIAN FULL COV WITH K FOLD ({nr_kfold_split} folds) ')
    algorithm_list = ['full_cov', 'diag_cov', 'tied_cov']
    nr_components_list=[1, 2, 4, 8, 16, 32]
    priors_res = {}
    for algorithm in algorithm_list: 
        print(f'\nGMM Algorithm {algorithm}')
        for prior_cl_T in application_priors:
            priors_res[str(prior_cl_T)] = []
            for nr_components in nr_components_list:
                min_dcf, _, _ = lib.K_fold(
                    D, 
                    L, 
                    GmmClassifier, 
                    z_norm=True, 
                    gaus=False, 
                    pca_m=None, 
                    k=nr_kfold_split, 
                    prior_cl_T=prior_cl_T, 
                    cfp=cfp, cfn=cfn, 
                    algorithm=algorithm, 
                    nr_clusters=nr_components
                )
                priors_res[str(prior_cl_T)].append(min_dcf)
                print(F"min DCF GMM {algorithm} with prior=%.1f and components={nr_components}:  %.3f\n" %(prior_cl_T, min_dcf))
        lib.plot_dcf_gmm(
            prior1_res=priors_res['0.5'], 
            prior2_res=priors_res['0.9'], 
            prior3_res=priors_res['0.1'], 
            components=nr_components_list, 
            model_name=f'{algorithm}'
        )


def actual_dcf(D, L, application_priors, nr_kfold_split, cfp, cfn):
    """
    ACTUAL DCF FOR BEST CLASSIFIERS
        - GMM Tied 4 components
        - Linear Logistic Regression
        - Linear SVM
    """
    PI_T = 0.5

    for PRIOR in application_priors:
        print(f'\n------------------ PRIOR {PRIOR}')
        ## GMM (4 comp) Gaussian Tied
        nr_components = 4
        algorithm = 'tied_cov'
        min_dcf_gmm, act_dcf_gmm, _ = lib.K_fold(
            D, 
            L, 
            GmmClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None,
            k=nr_kfold_split, 
            prior_cl_T=PRIOR, 
            cfp=cfp, 
            cfn=cfn, 
            algorithm=algorithm, 
            nr_clusters=nr_components,
            actual_dcf=True,
            plot_bayes_error=True,
            model_name_error_plot=f'GMM_{algorithm}_{nr_components}_comp',
        )
        print(F"min DCF GMM {algorithm} with prior={PRIOR:.1f} and components={nr_components}:  {min_dcf_gmm:.3f}")
        print(F"act DCF GMM {algorithm} with prior={PRIOR:.1f} and components={nr_components}:  {act_dcf_gmm:.3f}\n")

        ## Linear Logistic Regression
        _l = 0 # lambda
        min_dcf_logreg, act_dcf_logreg, _ = lib.K_fold(
            D, 
            L, 
            LogisticRegressionClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None,
            k=nr_kfold_split, 
            prior_cl_T=PRIOR, 
            cfp=cfp, 
            cfn=cfn, 
            _lambda=_l, 
            regularized=True, 
            pi_T=PI_T,
            actual_dcf=True,
            plot_bayes_error=True,
            model_name_error_plot=f'LINEAR_LOG_REG_lambda_{_l}',
        )
        print(f"min DCF LOGISTIC REGRESSION REG by {PI_T} with prior={PRIOR} and lambda={_l}:  {min_dcf_logreg:.3f}")
        print(f"act DCF LOGISTIC REGRESSION REG by {PI_T} with prior={PRIOR} and lambda={_l}:  {act_dcf_logreg:.3f}\n")

        ## Linear SVM
        C = 1
        min_dcf_linsvm, act_dcf_linsvm, _ = lib.K_fold(
            D, 
            L, 
            SVMLinearClassifier, 
            z_norm=True, 
            gaus=False, 
            pca_m=None,
            k=nr_kfold_split,
            prior_cl_T=PRIOR, 
            cfp=cfp, 
            cfn=cfn, 
            C=C, 
            rebalanced=True, 
            pi_T=PI_T,
            actual_dcf=True,
            plot_bayes_error=True,
            model_name_error_plot=f'LINEAR_SVM_C_{C}',
        )
        print(f"min DCF LINEAR SVM with prior={PRIOR:.1f} and C={C}:  {min_dcf_linsvm:.3f}")
        print(f"act DCF LINEAR SVM with prior={PRIOR:.1f} and C={C}:  {act_dcf_linsvm:.3f}\n") 

def score_calibration(D, L, classifier:'BaseClassifier', nr_kfold_split, cfp, cfn, model_name='model', **classifier_kwargs):
    PRIOR = 0.5
    _lambda_log_reg = 0
    min_dcf, act_dcf, klass = lib.K_fold(
        D, 
        L, 
        classifier_class=classifier, 
        z_norm=True, 
        gaus=False, 
        pca_m=None, 
        k=nr_kfold_split, 
        prior_cl_T=PRIOR, 
        cfp=cfp, 
        cfn=cfn,
        actual_dcf=True,
        plot_bayes_error=False,
        **classifier_kwargs
    )

    # np.save('data/scores', klass.scores)
    # np.save('data/labels', klass.L)

    print('first set', (min_dcf, act_dcf))

    # saved_scores = np.load('data/scores.npy')
    # saved_labels = np.load('data/labels.npy')

    ############################
    ## recalibrate score with k fold on scores using Logistic Regression
    min_dcf_reb, act_dcf_reb, klass_reb = lib.K_fold(
        D=lib.rowv(klass.scores), 
        L=klass.L, 
        classifier_class=LogisticRegressionClassifier, 
        actual_dcf=True, 
        z_norm=True, 
        gaus=False, 
        pca_m=None, 
        k=nr_kfold_split, 
        prior_cl_T=0.5, 
        cfp=cfp, 
        cfn=cfn,
        _lambda=_lambda_log_reg, 
        regularized=True, 
        pi_T=0.5,
    )
    print('second set', (min_dcf_reb, act_dcf_reb))
    lib.bayes_error_plot(
        scores=klass.scores,
        labels=klass.L,
        model_name=f'{model_name}_regularized',
        scores_rebalanced=klass_reb.scores,
        labels_rebalanced=klass_reb.L
    )
    


def _save_evaluation_scores(scores, file_name):
    if not os.path.exists(f'data/{file_name}.npy'):
        np.save(f'data/{file_name}', scores)


def evaluation(DTR, LTR, cfp, cfn):
    print('********  STARTING EVALUATION ***********\n')
    target_prior = 0.5
    GMM_scores_file_name = 'evaluation_scores_GMM_tied_4'
    LOGREG_scores_file_name = 'evaluation_scores_LOGREG'
    LSVM_not_cal_scores_file_name = 'evaluation_scores_SVM_not_calibrated'
    LSVM_cal_scores_file_name = 'evaluation_scores_SVM_calibrated'

    DTE, LTE = lib.load_binary_data(TEST_DATA_FILE, NR_FEATURES)

    
    # APPLYING Z-NORMALIZATION
    DTR, DTE = lib.z_normalization(DTR, DTE)

    for prior_cl_T in application_priors:
        print(f'\n------------------------------- PRIOR {prior_cl_T}')

        ### GMM TIED (4) CLASSIFIER
        nr_clusters = 4
        algorithm = 'tied_cov'
        gmm = GmmClassifier(DTR, LTR)
        gmm.compute_score(DTE, LTE, nr_clusters=nr_clusters, algorithm=algorithm)

        min_dcf = lib.compute_min_DCF(
            scores=gmm.scores,
            labels=LTE,
            prior_cl1=prior_cl_T,
            Cfn=cfn,
            Cfp=cfp
        )

        act_dcf = lib.compute_act_DCF(
            scores=gmm.scores,
            labels=LTE,
            prior_cl1=prior_cl_T,
            Cfn=cfn,
            Cfp=cfp
        )
        if prior_cl_T == target_prior:
            _save_evaluation_scores(gmm.scores, GMM_scores_file_name)
        print(f'EVALUATION: GMM TIED (4) prior {prior_cl_T}, nr_comp. {nr_clusters}, min DCF {min_dcf}, act DCF {act_dcf}\n')

        for pi_T in application_priors:
            print(f'pi_TR {pi_T}')

            ### LOG REG CLASSIFIER
            _lambda = 0
            lr = LogisticRegressionClassifier(DTR, LTR)
            lr.compute_score(DTE, LTE, _lambda=_lambda, regularized=True, pi_T=pi_T)

            min_dcf = lib.compute_min_DCF(
                scores=lr.scores,
                labels=LTE,
                prior_cl1=prior_cl_T,
                Cfn=cfn,
                Cfp=cfp
            )

            act_dcf = lib.compute_act_DCF(
                scores=lr.scores,
                labels=LTE,
                prior_cl1=prior_cl_T,
                Cfn=cfn,
                Cfp=cfp
            )
            if prior_cl_T == target_prior == pi_T:
                _save_evaluation_scores(lr.scores, LOGREG_scores_file_name)
            print(f'EVALUATION: LOGISTIC REGRESSION prior {prior_cl_T}, pi_T {pi_T}, lambda {_lambda} min DCF {min_dcf}, act DCF {act_dcf}\n')


            ## LINEAR SVM CLASSIFIER
            C = 1
            lsvm = SVMLinearClassifier(DTR, LTR)
            lsvm.compute_score(DTE, LTE, C=C, rebalanced=True, pi_T=pi_T)

            min_dcf = lib.compute_min_DCF(
                scores=lsvm.scores,
                labels=LTE,
                prior_cl1=prior_cl_T,
                Cfn=cfn,
                Cfp=cfp
            )

            act_dcf = lib.compute_act_DCF(
                scores=lsvm.scores,
                labels=LTE,
                prior_cl1=prior_cl_T,
                Cfn=cfn,
                Cfp=cfp
            )
            if prior_cl_T == target_prior == pi_T:
                _save_evaluation_scores(lsvm.scores, LSVM_not_cal_scores_file_name)
            print(f'EVALUATION: LINEAR SVM prior {prior_cl_T}, pi_T {pi_T}, C {C}, min DCF {min_dcf}, act DCF {act_dcf}\n')
            # recalibrate linear SVM
            prior_lr_recalibrator = 0.5
            min_dcf_reb, act_dcf_reb, klass_reb = lib.K_fold(
                D=lib.rowv(lsvm.scores), # scores of SVM as data input for LR
                L=LTE, 
                classifier_class=LogisticRegressionClassifier, 
                actual_dcf=True, 
                z_norm=True, 
                gaus=False, 
                pca_m=None, 
                k=nr_kfold_split, 
                prior_cl_T=prior_lr_recalibrator, 
                cfp=cfp, 
                cfn=cfn,
                _lambda=0, 
                regularized=False, 
            )
            if prior_cl_T == target_prior == pi_T:
                _save_evaluation_scores(klass_reb.scores, LSVM_cal_scores_file_name)
                _save_evaluation_scores(klass_reb.L, LSVM_cal_scores_file_name+'_labels')
            print(f'EVALUATION: LINEAR SVM (CALIBRATED) prior DVM {prior_cl_T}, prior LR recalibrate {prior_lr_recalibrator}, pi_T {pi_T}, C {C}, min DCF {min_dcf_reb}, act DCF {act_dcf_reb}\n')

    GMM_scores = np.load(f'data/{GMM_scores_file_name}.npy')
    LOGREG_scores = np.load(f'data/{LOGREG_scores_file_name}.npy')
    LSVM_not_cal_scores = np.load(f'data/{LSVM_not_cal_scores_file_name}.npy')
    LSVM_cal_scores = np.load(f'data/{LSVM_cal_scores_file_name}.npy')
    LSVM_cal_scores_labels = np.load(f'data/{LSVM_cal_scores_file_name}_labels.npy')

    lib.bayes_error_plot_multiple(
        scores_list=[GMM_scores, LOGREG_scores, LSVM_not_cal_scores],
        labels=LTE,
        model_name_list=[GMM_scores_file_name, LOGREG_scores_file_name, LSVM_not_cal_scores_file_name]
    )


    lib.bayes_error_plot_multiple(
        scores_list=[GMM_scores, LOGREG_scores],
        labels=LTE,
        model_name_list=[GMM_scores_file_name, LOGREG_scores_file_name],
        scores_rebalanced=LSVM_cal_scores,
        labels_rebalanced=LSVM_cal_scores_labels,
        name_rebalanced=LSVM_cal_scores_file_name,
    )

    lib.ROC_plot_binary(
        scores_list=[GMM_scores, LOGREG_scores, LSVM_not_cal_scores],
        labels=LTE,
        model_name_list=[GMM_scores_file_name, LOGREG_scores_file_name, LSVM_not_cal_scores_file_name]

    )



if __name__ == '__main__':
    import app.libs.ml_lib as lib
    from app.core.classifiers import (
        BaseClassifier,
        GaussianClassifier, 
        GaussianBayesianClassifier, 
        GaussianTiedClassifier,
        LogisticRegressionClassifier,
        SVMLinearClassifier,
        SVMKernelRBFClassifier,
        SVMKernelPolynomialClassifier,
        GmmClassifier,
    )
    print('\t#\t# USING DATASET ---- ', PROVAAAAAAAAA, '\n')

    D, L = lib.load_binary_data(TRAINING_DATA_FILE, NR_FEATURES)

    # D, L = lib.load_iris_binary()
    # D, L = lib.load_iris_binary_reduced(5)
    application_priors = [0.5, 0.9, 0.1]
    # D_norm, _ = lib.z_normalization(D)
    # D_norm_gau = lib.gaussianization(D_norm, D_norm)

    nr_kfold_split = 4
    cfp = 1
    cfn = 1


    # plotting(D, L)

    # gaussian(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # linear_logistic_regression(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # svm_linear(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # svm_kernel_rbf(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # svm_kernel_polynomial(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # gmm(D, L, application_priors, nr_kfold_split, cfp, cfn)

    # actual_dcf(D, L, application_priors, nr_kfold_split, cfp, cfn)


    # score_calibrate_SVM = partial(score_calibration, classifier=SVMLinearClassifier)
    # score_calibrate_SVM(
    #     D=D, L=L, 
    #     nr_kfold_split=nr_kfold_split, 
    #     cfp=cfp, cfn=cfn, 
    #     model_name='LINEAR_SVM_C_1',
    # )
    
    # score_calibrate_GMM = partial(score_calibration, classifier=GmmClassifier)
    # score_calibrate_GMM(
    #     D=D, 
    #     L=L, 
    #     nr_kfold_split=nr_kfold_split, 
    #     cfp=cfp, cfn=cfn, 
    #     model_name='GMM_tied_cov_4_comp', 
    #     algorithm='tied_cov', 
    #     nr_clusters=4,
    # )

    # score_calibrate_LOGREG = partial(score_calibration, classifier=LogisticRegressionClassifier)
    # score_calibrate_LOGREG(
    #     D=D, 
    #     L=L, 
    #     nr_kfold_split=nr_kfold_split, 
    #     cfp=cfp, 
    #     cfn=cfn, 
    #     model_name='LINEAR_LOG_REG_lambda_0', 
    #     _lambda=0,
    # )

    evaluation(DTR=D, LTR=L, cfp=cfp, cfn=cfn)








