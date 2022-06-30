from ast import Yield
from typing import Tuple
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn

from app.core.classifiers import BaseClassifier

def load_file(path: str):
    with open(path, 'r') as f:
        return list(csv.reader(f, delimiter=','))


def load_iris() -> tuple:
    """
    data.shape = (150,4)
    Tuple[data, labels]
    """
    return datasets.load_iris()['data'], datasets.load_iris()['target']


def load_iris_beautify() -> tuple:
    """
    Traspose the data from (150, 4) --> (4, 150)
    Tuple[data.T, labels]
    """
    return datasets.load_iris()['data'].T, datasets.load_iris()['target']


def load_iris_reduced() -> tuple:

    data_reduced, labels_reduced = load_iris()
    # data_reduced = all columns, 1 every 10 rows
    # labels_reduced = 1 every 10 column (1-d array)
    return data_reduced[::10, :], labels_reduced[::10]


def load_iris_reduced_beautify() -> tuple:
    data_reduced, labels_reduced = load_iris_beautify()
    # data_reduced = all columns, 1 every 10 rows
    # labels_reduced = 1 every 10 column (1-d array)
    return data_reduced[:, ::10], labels_reduced[::10]

def load_iris_binary():
    """
    Returns only  iris virginica and iris versicolor samples
    D.shape = (4, 100)
    L.shape = (100,)
    """
    D, L = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def load_iris_binary_reduced() -> tuple:

    data_reduced, labels_reduced = load_iris_binary()
    # data_reduced = all columns, 1 every 10 rows
    # labels_reduced = 1 every 10 column (1-d array)
    return data_reduced[:, ::10], labels_reduced[::10]


def colv(arr):
    '''
    get a 1-d array and return a column vector
    '''
    return arr.reshape(-1, 1)
    # oppure return arr.reshape((arr.size, 1))


def rowv(arr):
    """
    get a 1-d array and return a row vector
    """
    return arr.reshape(1, -1)


def flatten(mat):
    """
    get a mat and return it as 1d array
    """
    return mat.ravel()


def load_solution(name: str, verbose=True):
    """
    automatically searche and load solution in path .solution/
    example
        my_lib.load_solution('lab05_Generative_Models/Solution/logMarginal_MVG')
    """
    res = np.load(f'{name}.npy')
    if verbose:
        print_nice(f'SOLUTION solution : "{name}"', res)
    return res


def logpdf_1sample(x, mu, C):
    """
    LAB04
    Compute the log density for one sample ata a time
    (not efficient, we need to call this in a loop through all
    column vectors x)
    Params
        :x: is a column vector
    Returns
        :1-d array
    """
    res = (x.shape[0]*0.5)*np.log(2*np.pi)  # -M/2*log(2pi)
    _, determinant = np.linalg.slogdet(C)  # _, log(|C|)
    res += 0.5*determinant  # -(1/2)*log(|C|)
    signma_inverse = np.linalg.inv(C)  # C^-1
    res += np.dot(
        0.5*(x-mu).T,
        np.dot(signma_inverse, (x-mu))
    )  # -(1/2) * (x-mu)^T * C^-1 * (x-mu)
    return flatten(np.negative(res))

#             n
#    ├───────────────── ├
#    │  │  |     |   |  |
#  M │  │  |  X  |   |  |  X = [M x n] of x col vec
#    │  │  |     |   |  |  (one col vec is x)
#    │  │  |     |   |  |
#    └──────────────────└
# iterate for all the x column vector passing them to the logpdf_1sample()


def logpdf_GAU_ND(X, mu, C):
    """
    LAB04

    Returns
        :1-d array
    """
    try:

        P = np.linalg.inv(C) # C^-1
    except:
        P = np.linalg.pinv(C) # C^-1
    # uses broadcast over the M vertical direction (X.shape[0])
    res = -0.5 * X.shape[0] * np.log(2*np.pi) + 0.5*np.linalg.slogdet(P)[1]
    res = res - 0.5 * ((X-mu)*np.dot(P, (X-mu))).sum(0)
    return res

#


def pdf_GAU_ND(X, mu, C):
    """
    LAB04

    Returns
        :1-d array exponential version
    """
    return np.exp(logpdf_GAU_ND(X, mu, C))


########## ----- ########
########## -------- ###########
# MAXIMUM LIKEHHOD
########## -------- ###########
########## ----- ########
def compute_empirical_cov(X):
    """
    The maximum likehood Covariance is the Empirical Covariance
    """
    mu = compute_empirical_mean(X)
    cov = np.dot((X - mu), (X-mu).T) / X.shape[1]
    return cov


def compute_empirical_mean(X):
    """
    The maximum likehood Mean is the Empirical Mean
    """
    return colv(X.mean(1))


def compute_empirical_mean_and_cov(X, is_bayesian=False) -> Tuple:
    """
    Same as before functions but in one time

    :is_bayesian: if True the covariance Matrix will have only 
                  the elements in the diagonal (lab05 II) because (only for Gaussian Classifier)
                  the covariance matrix of Bayesian is the same as Gaussian but with only diagonal elements
    Returns:
        mu
        C
    """
    mu = colv(X.mean(1))
    C = np.dot((X - mu), (X-mu).T) / X.shape[1]

    if is_bayesian:
        # to make the C matric diagonal multiply it for an identity matrix of same shape
        C = C * np.identity(C.shape[0])
    return (mu, C)


def within_class_covariance(data: np.array, whole_labels: np.array, labels: list):
    """
    Lab05 II (formula in the Tied Cov Gaussian Classifier) 
    Given the data, the Classes labels, returns the Within Class Covariance

    :data:   must be the whole dataset, not splitted for class labels
    :whole_labels: a 1-d np array that lists all classes labels
    :labels: a list with only labels identifier (like 0,1,2 for the setosa dataset)
    """
    SW = 0
    for label in labels:
        data_for_label = data[:, whole_labels == label]
        C = compute_empirical_cov(data_for_label)
        # (n_c -> nr of elment for each Class Label)
        n_c = (whole_labels == label).sum()
        SW += n_c * C
    return SW/data.shape[1]


def loglikehood(X, mu, C):
    """
    LAB04
    (lez 11 - 28:00) is the sum of the log density for all the samples
    so we exploit the log density function and sum over it
    """
    return logpdf_GAU_ND(X, mu, C).sum()


def likehood(X, mu, C):
    """
    LAB04 (probabily not useful for underflow)
    simply the exponential of the log-likehood (lab4 31.23)
    """
    Y = np.exp(logpdf_GAU_ND(X, mu, C))
    # the product of the densisty for all the vectors
    return Y.prod()


def print_nice(descr: str, data=None):
    type_ = str(type(data)) if data is not None else ''
    print(descr + ' | type = ' + type_, data, sep='\n', end='\n\n')


def split_db_2to1(D, L, seed=0):
    """
    Lab05
    Prof function to randomly split a Dataset D and its Labels L
    in Training and Test Data

    Returns: (DTR, LTR), (DTE, LTE)
        DTR = Training Data 
        LTR = Training Labels

        DTE = Test Data
        LTE = Test Labels
    """
    nTrain = int(D.shape[1]*2.0/3.0)

    # seed to always generate the same random values
    np.random.seed(seed)

    # D.shape[1] random indexes from 0 to D.shape[1]-1
    idx = np.random.permutation(D.shape[1])
    # train indexes from 0 to nTrain
    idxTrain = idx[0:nTrain]
    # test indexes the remaining
    idxTest = idx[nTrain:]
    # takes only idxTrain columns..
    DTR = D[:, idxTrain]
    # ..takes only idxTest columns..
    DTE = D[:, idxTest]
    # .. same for 1-d array
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

######################
######################
######################
def load_binary_train_data(fname, nr_features):
    DList = []
    labelsList = []
    hLabels = {}

    # for i in range(nr_features):
    #     hLabels[f'feature_{i+1}'] = i

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:nr_features]
                attrs = colv(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def _setup_data_plot(D, L, nr_features, positive_c_name, negative_c_name):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    features_dict = {}

    for i in range(nr_features):
        features_dict[i] = f'feature_{i}'

    return D0, D1, features_dict

def plot_hist_binary(D, L, nr_features, positive_c_name, negative_c_name, file_name=''):

    D0, D1, features_dict = _setup_data_plot(D, L, nr_features, positive_c_name, negative_c_name)

    for dIdx in range(nr_features):
        plt.figure()
        plt.xlabel(features_dict[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, ec="#090b33", label = negative_c_name)
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, ec="#452b0d", label = positive_c_name)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/histogram/hist_%s_%d.pdf' % (file_name, dIdx))
    plt.show()

def plot_scatter(D, L, nr_features, positive_c_name, negative_c_name):
    
    D0, D1, features_dict = _setup_data_plot(D, L, nr_features, positive_c_name, negative_c_name)

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(features_dict[dIdx1])
            plt.ylabel(features_dict[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = negative_c_name)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = positive_c_name)
        
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/scatter/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def plot_pearson_heatmap(D, L):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.savefig('plots/pearson/Pearson_all.pdf')
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Blues", square=True,cbar=False)
    plt.savefig('plots/pearson/Pearson_Male.pdf')
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Oranges", square=True, cbar=False)
    plt.savefig('plots/pearson/Pearson_Female.pdf')

def gaussanization(DTR, DTE):
    """
    DTE can also be the same of DTR in case of training
    use DTE==DTE when ranking test data over training data
    """
    gaussianized_data = np.zeros(DTE.shape)
    for i in range (DTR.shape[0]):
        ranks_for_row = []
        for j in range(DTE.shape[1]):
            tot = 0 
            line = DTR[i, :]
            value = DTE[i,j]
            tot += (line<value).sum()
            tot += 1
            rank = tot/(DTE.shape[1] + 2)
            ranks_for_row.append(rank)
        gaussianized_data[i,:] = np.asarray(ranks_for_row)

    return sp.stats.norm.ppf(gaussianized_data)

def z_normalization(D):
    # subtract dataset mean from each sample and divide by standard deviation
    return sp.stats.zscore(D, axis=1)

def PCA(data, m:int):
    """
    :m = nr of features to keep
    """
    print(f'Applying PCA with reduction of {m}')
    # mean lung l'asse x
    mu = data.mean(1)  # 1-d array

    ########## ----- ########
    ########## -------- ###########
    # mean to col vector
    ########## -------- ###########
    ########## ----- ########
    mu = colv(mu)

    ########## ----- ########
    ########## -------- ###########
    # subtract mean on each line along X axes
    ########## -------- ###########
    ########## ----- ########
    DC = data - mu  # DC = matrix of centered data

    ########## ----- ########
    ########## -------- ###########
    # DC covariance matrix = 1/N * Dc* Dc^T
    ########## -------- ###########
    ########## ----- ########
    DCcov = DC.dot(DC.T)/DC.shape[1]  # simetric with respect to principal diagonal
    # or directly wit .cov numpy function
    # DCcov = np.cov(DC, bias=True)

    ########## ----- ########
    ########## -------- ###########
    # compute DCcov eigen-val and eigen-vec
    # For a generic square matrix we can use the library function numpy.linalg.eig .
    # Since the covariance matrix is symmetric, we can use the function numpy.linalg.eigh
    ########## -------- ###########
    ########## ----- ########
    # eigen value not sorted
    s, U = np.linalg.eigh(DCcov)  # s = eigen-val sorted asc, U = eigen-vec
    # s =     [0.02367693   0.07768784   0.24105279   4.2000546 ]
    # U = [
    #     [ 0.3154882   0.58203006  0.65658814 -0.36138648]
    #     [-0.3197243  -0.5979094   0.7301621   0.08452249]
    #     [-0.4798389  -0.07623544 -0.1733726  -0.85667074]
    #     [ 0.75365657 -0.5458329  -0.07548022 -0.35828903]
    # ]


    ########## ----- ########
    ########## -------- ###########
    # let's take the eigen vector corresponding to the m largest eigen values
    # let's for example fix  m = 2 (columns are 4 so we are reducing the dimensionality by 2)
    # We want the eigen vector ordered in descending with respect to their corresponding
    # eigen values, than we take only the first m colum only
    # after we reversed the matrix in the sense of columns
    ########## -------- ###########
    ########## ----- ########
    # [reverse the order of each column of the eigen vectors][take the first m (2)]
    P = U[:, ::-1][:, :m]
    # U = [
    #     [-0.36138648   0.65658814 ]
    #     [ 0.08452249   0.7301621  ]
    #     [-0.85667074  -0.1733726  ]
    #     [-0.35828903  -0.07548022 ]
    # ]

    ########## ----- ########
    ########## -------- ###########
    # let's apply the PROJECTION of all dataset point
    # yi = P^T * xi
    ########## -------- ###########
    ########## ----- ########
    # P.t.shape = (2, 4), data.shape = (4, 150)
    DProjList = np.dot(P.T, data)  # shape (2, 150)
    return DProjList

def spilt_K_fold(D, L, k:int, seed=0) -> Yield(Tuple):
    """
        Yields: (DTR, LTR, DVA, LVA) for each of the k folds
            DTR = Data Training
            LTR = Labels Training

            DVA = Data Validation
            LVA = Labels Validation

            i.e. with k == 4
            for i in range(k)
                i == 0)
                    D = [[ D_fold_1 D_fold_2 D_fold_2 D_fold_4 ]] 
                    L =  [ L_fold_1 L_fold_2 L_fold_2 L_fold_4 ]
                                        
                                        DTR         LTR       DVA       LVA
                        --> yields (D_fold_234, L_fold_234, D_fold_1, L_fold_1)

                i == 1)
                    D = [[ D_fold_1 D_fold_2 D_fold_2 D_fold_4 ]] 
                    L =  [ L_fold_1 L_fold_2 L_fold_2 L_fold_4 ]
                                    
                                        DTR         LTR       DVA       LVA
                        --> yields (D_fold_134, L_fold_134, D_fold_2, L_fold_2)
            ...
    """
    if k < 2:
        raise ValueError("nr of folds must be at least 2")

    # seed to always generate the same random values
    np.random.seed(seed)

    # SHUFFLE columns ids
    # D.shape[1] (tot nr of samples) random indexes from 0 to D.shape[1]-1
    idx = np.random.permutation(D.shape[1])
    print('\t\t\t WARNING RECOMPUTING RANDOMNESS IN K FOLD SPLIT ---------------------')

    # how many sample in each fold
    nr_samples_in_folds = int(D.shape[1] / k)

    
    for i in range(k):
        #### take the Validation ids
        start_index = i * nr_samples_in_folds
        end_index   = (i+1) * nr_samples_in_folds
        # at last iteration tak all the remaining samples
        if i == k-1:
            end_index = None 
        # take the i portion of random column indices
        import ipdb; ipdb.set_trace()
        idx_validation = idx[start_index : end_index]

        #### removes the validation indexes from the dataset
        #### to obtain only the Training samples and labels
        DTR = np.delete(D, idx_validation, axis=1)
        LTR = np.delete(L, idx_validation)

        DVA = D[:,idx_validation]
        LVA = L[idx_validation]

        yield (DTR, LTR, DVA, LVA)



def K_fold(D, L, classifier_class: BaseClassifier, k:int, seed=0):

    for i in range(k):

        fold_generator = spilt_K_fold(D, L, k, seed)
        DTR, LTR, DVA, LVA = next(fold_generator)
        import ipdb; ipdb.set_trace()
        classifier: BaseClassifier = classifier_class(DTR, LTR)
        classes_prior = [0.5, 0.5]
        classifier.train(DTE=DVA, classes_prior=classes_prior)
        aaa = classifier.classify(LTE=LVA)
        import ipdb; ipdb.set_trace()
        aaa






# # XXX main
# if __name__ == '__main__':
#     data, data_labels = load_iris()

#     data = np.array(data) # matrix put inside min list in column
#     print('raw data', data, sep='\n', end='\n\n')
#     print(data.size, data.shape)

