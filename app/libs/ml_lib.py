from typing import Tuple
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn

iris = datasets.load_iris()

label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}


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


def logpdf_GAU_ND_bad(X, mu, C):
    """
    LAB04
    Compute the log density looping through all the column
    vectors calling the function `logpdf_1sample' that computes
    the log density for one column vector
    :X: all column vectors (one is col vector is x)
    :mu: column vectors of means
    :C: Covariance matrix
    """
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.size[1])]
    return flatten(np.array(Y))


def logpdf_GAU_ND_good(X, mu, C):
    """
    LAB04

    """
    try:
        P = np.linalg.inv(C) # C^-1
    except:
        P = np.linalg.pinv(C) # C^-1
    # First compute the constatns
    const = -0.5 * X.shape[0] * np.log(2 * np.pi)  # -M/2*log(2pi)
    _, determinant = np.linalg.slogdet(C)  # |C|
    const += - 0.5 * determinant  # -(1/2)*log(|C|)
    # Then loop for all the x column vectors
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    return np.array(Y).ravel()


# optimized hided by prof
def logpdf_GAU_ND_perfect(X, mu, C):
    """
    LAB04

    Returns
        :1-d array
    """
    try:

        P = np.linalg.inv(C) # C^-1
    except:
        P = np.linalg.pinv(C) # C^-1
    # uses broadcast i guess over the M vertical direction (X.shape[0])
    res = -0.5 * X.shape[0] * np.log(2*np.pi) + 0.5*np.linalg.slogdet(P)[1]
    res = res - 0.5 * ((X-mu)*np.dot(P, (X-mu))).sum(0)
    return res

#


def pdf_GAU_ND_perfect(X, mu, C):
    """
    LAB04

    Returns
        :1-d array exponential version
    """
    return np.exp(logpdf_GAU_ND_perfect(X, mu, C))


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
    return logpdf_GAU_ND_perfect(X, mu, C).sum()


def likehood(X, mu, C):
    """
    LAB04 (probabily not useful for underflow)
    simply the exponential of the log-likehood (lab4 31.23)
    """
    Y = np.exp(logpdf_GAU_ND_perfect(X, mu, C))
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
def load_bniary_train_data(fname, nr_features):
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

def plot_hist_binary(D, L, nr_features, positive_c_name, negative_c_name):

    D0, D1, features_dict = _setup_data_plot(D, L, nr_features, positive_c_name, negative_c_name)

    for dIdx in range(nr_features):
        plt.figure()
        plt.xlabel(features_dict[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, ec="#090b33", label = negative_c_name)
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, ec="#452b0d", label = positive_c_name)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/histogram/hist_%d.pdf' % dIdx)
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



# # XXX main
# if __name__ == '__main__':
#     data, data_labels = load_iris()

#     data = np.array(data) # matrix put inside min list in column
#     print('raw data', data, sep='\n', end='\n\n')
#     print(data.size, data.shape)

