from abc import ABC, abstractmethod, abstractclassmethod
import sys
from typing import Literal
from xml.dom import ValidationErr
sys.path.append('.')
import app.libs.ml_lib as lib
import scipy as sp
import numpy as np
import os
import pylab


class BaseClassifier(ABC):

    @abstractmethod
    def __init__(self, D, L):
        self.D = D
        self.L = L

    @abstractmethod
    def compute_score(self, DTE, LTE=None, **classifiers_kwargs):
        """Returns scores"""
        raise NotImplementedError

    @abstractmethod
    def train(self, DTE, classes_prior: list):
        """Returns posteriors"""
        raise NotImplementedError

    @abstractmethod
    def classify(self, LTE):
        raise NotImplementedError

class GaussianClassifier(BaseClassifier):
    """
    >>> score
    >>> train
    >>> classify

    """

    def __init__(self, D, L, is_tied=False):
        super(GaussianClassifier, self).__init__(
            D,
            L
        )
        self.is_tied = is_tied
        self.is_bayesian = False
        tied_cov = None
        if self.is_tied:
            tied_cov = lib.within_class_covariance(data=self.D, whole_labels=self.L, labels=set(self.L.ravel())) 

        mean, cov = lib.compute_empirical_mean_and_cov(
                X=self.D,
                is_bayesian=self.is_bayesian,
                is_tied=self.is_tied,
                tied_cov=tied_cov
        )
        self.mean = mean
        self.cov  = cov
        self.posteriors = None
        self.predicted_labels = None

    def _log_domain_scores(self, class_gaussian_param: dict, class_labels: list, DTE):
        ########
        # Calculate the Conditional Density for each Class
        # CLASS CONDITIONAL DENSITY (Matrix of SCORES)
        ########
        logSJoint = np.zeros((len(class_labels), DTE.shape[1]))
        class_cond_density = []
        for el in class_labels:
            # Dtest_for_class = DTE[:,LTE==el]
            class_cond_density.append(
                lib.logpdf_GAU_ND(DTE, *class_gaussian_param[el]))
        # make a unique matrix of scores withe these arrays
        S = np.vstack(class_cond_density)
        return S

    def _log_domain_posterior(self, Scores, class_priors: list):
        ########
        # Calculate the JOINT PROBABILITY MATRIX
        # log domain so we sum each element of the 'Class Conditional Density
        # Matrices' for its Prior Probability
        ########
        # transform to a column vector to broadcast the multiplication to all 3x50 elements
        classes_Prior_Prob = lib.colv(np.array(class_priors))

        logSJoint = Scores + np.log(classes_Prior_Prob)

        ########
        # Calculate the MARGINAL
        # the sum of the joint along all classes of each sample
        # with scipy special sum function
        ########
        log_marginal = sp.special.logsumexp(logSJoint, axis=0)


        ########
        # Calculate the POSTERIOR PROBABILITY
        # as the difference (nota ratio cause in log domain)
        # between each el of Joint Matrix and the
        # respective MARGINAL value
        ########
        log_Posterior = logSJoint - lib.rowv(log_marginal)

        # return to exp
        log_Posterior_to_exp = np.exp(log_Posterior)
        return log_Posterior_to_exp


    def compute_score(self, DTE, LTE=None, **classifiers_kwargs):
        """
        Computes
            self.scores
            self.llr
        """
        ########
        # Split the DTR in its classes and
        # for each of the Classes compute empirical
        # Mean and empirical Variance
        ##
        # class_gaussian_parameters = {
        # <class_index> : (
        # <class_empirical_mean>, <class_empirical_cov>
        # )
        # }
        ########
        class_gaussian_parameters = {}
        class_labels = set(self.L.ravel()) # single classes values
        for el in class_labels:
            D_train_for_class = self.D[:, self.L == el]
            class_gaussian_parameters[el] = lib.compute_empirical_mean_and_cov(
                X=D_train_for_class,
                is_bayesian=self.is_bayesian,
                is_tied=self.is_tied,
                tied_cov=self.cov
            )

        scores = self._log_domain_scores(
            class_gaussian_param=class_gaussian_parameters,
            class_labels=class_labels,
            DTE=DTE,
        )
        self.scores = scores[1] - scores[0] # LLR
        return self.scores
        


    def train(self, classes_prior: list):
        """
        computes the posteriors
        to be called after:
            >>> def score()
        """
        if self.scores is None:
            raise ValidationErr ("First you have to score the model")

        self.posteriors = self._log_domain_posterior(
            Scores=self.scores,
            class_priors=classes_prior
        )


    def classify(self, LTE):
        """ 
        Computes the error rate
        to be called after:
            >>> def score()
            >>> def train()
        """
        print("Classifying with Gaussian Classifier")
        if self.posteriors is None:
            raise ValidationErr ("First you have to train the model")
    
        ########
        # Calculate the Max between calculated probability along Y axes
        # compare the indeces of the maximum value that corresponds to the
        # class and compare it with the correct test labels (LTE)
        ########
        # # # # # # # # # # # # # # # predicted_classes = self.posteriors.argmax(0)
        # # # # # # # # # # # # # # # prediction_result = predicted_classes == LTE

        # # # # # # # # # # # # # # # # nr correct prediction(sum all true values) / nr tot elements
        # # # # # # # # # # # # # # # precision = prediction_result.sum() / prediction_result.size
        self.predicted_labels = np.argmax(self.posteriors, axis=0)
        nr_correct_predictions = np.array(self.predicted_labels == LTE).sum()
        accuracy = nr_correct_predictions/LTE.size*100
        error_rate = 100-accuracy
        return error_rate

class GaussianBayesianClassifier(GaussianClassifier):
    def __init__(self, D, L):
        super(GaussianBayesianClassifier, self).__init__(
            D,
            L
        )
        self.is_bayesian = True

class GaussianTiedClassifier(GaussianClassifier):
    def __init__(self, D, L):
        super(GaussianTiedClassifier, self).__init__(
            D,
            L,
            is_tied = True
        )

class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self, D, L):
        super().__init__(
            D,
            L
        )
        self.scores = None

    def logreg_obj_wrap(self, DTR, LTR, _lambda, regularized=False, pi_T=None):
        def logreg_obj(v):
            """
            J(w, b) = (l/2)||w||^2 + (1/n)SUM(1 to n){ log(1 + exp^(-zi(w^Txi + b))) }

            if regularized == True (i.e. unbalanced data), LR scores will be balanced
            with classes prior and nr of elements (07_slide30)
            """
            # dimension of features space (number of features)
            M = DTR.shape[0]
            # transform label 0 to -1 and label 1 to 1
            Z = LTR * 2.0 - 1.0

            # divide the vector 'v' that contains both w than b (as last element)
            # and make it a column vector
            w = lib.colv(v[0 : M])
            b = v[-1]
            # since DTR = [x1, x2, ..., xn], to compute [w.T*x1, ..., w.T*xn]
            # we can avoid the for loop to compute """w^Txi"""  and then broadcast the bais '+ b'
            S = np.dot(w.T, DTR) + b

            if regularized: # (07_slide30)
            #                         """(priorTrue/nrTrue)SUM{ log(1 + exp^(-zi(w^TxTrue + b))) }"""
                cross_entropy_pos2 = (
                    (np.logaddexp(0, -S[:,Z==1] * Z[Z==1]) * pi_T) / (Z==1).sum()
                ).sum()
            #                         """(priorFalse/nrFalse)SUM{ log(1 + exp^(-zi(w^TxFalse + b))) }"""
                cross_entropy_neg2 = (
                    (np.logaddexp(0, -S[:,Z==-1] * Z[Z==-1]) * (1-pi_T)) / (Z==-1).sum()
                ).sum()
                res = _lambda * 0.5 * np.linalg.norm(w)**2 + cross_entropy_pos2 + cross_entropy_neg2
                return res
            # now compute          """(1/n)SUM(1 to n){ log(1 + exp^(-zi(w^Txi + b))) }"""
            # with the previous res """(1/n)SUM(1 to n){ log(1 + exp^(-zi(    S    ))) }"""
            # the log part is the same of log(exp^0 + exp^(-zi(    S    )))"""
            # so we should pass to this func an array of zeros and our -S*Z
            # the func do the broadcast so we can only pass a string 0
            # result will be a vector of this computation for each x element
            # for the 1/n part just compute the mean() of this sum
            cross_entropy = np.logaddexp(0, -S * Z).mean()

            # last part: cross_entropy + (l/2)||w||^2
            return       cross_entropy + _lambda * 0.5 * np.linalg.norm(w)**2

        return logreg_obj

    def compute_score(self, DTE, LTE=None, _lambda=None, pi_T=None, regularized=False):
        """Returns scores"""
        if regularized and pi_T is None:
            raise ValidationErr("Can't ask regularization without providing an empirical prior")
        _v, _J, _d = sp.optimize.fmin_l_bfgs_b(
            self.logreg_obj_wrap(DTR=self.D, LTR=self.L, _lambda=_lambda, pi_T=pi_T, regularized=regularized),
            np.zeros(self.D.shape[0] + 1),
            approx_grad=True
        )

        # recover the w values inside the v vector
        _w = _v[0 : self.D.shape[0]]
        # recover the b value inside the v vector
        _b = _v[-1]
        # scores from test sample
        self.scores = np.dot(_w.T, DTE) + _b
        return self.scores

    def train(self, DTE, classes_prior: list):
        """Returns posteriors"""
        raise NotImplementedError

    def classify(self, LTE):
        raise NotImplementedError

class SVMLinearClassifier(BaseClassifier):

    def __init__(self, D, L):
        super().__init__(
            D,
            L
        )
        self.scores = None
        self.C = None
        self.K = None
        self.gamma = None


    def _train_SVM(self, DTR, LTR, C, H, rebalanced:bool=False, pi_T=0.5):
        """
        :DTR data training
        :LTR label training
        :C Costs
        :H matrix H
        :rebalanced if True rebalance computation of classes scores by pi_T
        :pi_T prior of Class T for rebalancing

        returns alphastar and _x (dual loss) for SVM with scipy
            numerical calculator 
        """
        tot_qt_samples = DTR.shape[1]
        # @ bounds of dual formulation:
        #     0 <= alfa_i <= C
        # @ if we are rebalancing for the different classes:
        #     0 <= alpha_i <= C_class_T # for sample class1
        #     0 <= alpha_i <= C_class_F # for sample class2
        #      # with C_class_x = C*(prior_class_x/empirical_prior_class_x)
        #            # with: empirical_prior_class_x = n_sample_class_x / n_all_samples
        if rebalanced:
            # nr_samples_class_x / tot_nr_samples
            empirical_prior_T = DTR[:, LTR==1].shape[1] / tot_qt_samples
            empirical_prior_F = DTR[:, LTR==0].shape[1] / tot_qt_samples
        
            C_T = C*((pi_T) / empirical_prior_T)
            C_F = C*((1 - pi_T) / empirical_prior_F)

            bounds = list(
                map(
                    lambda x: (0, C_T) if x==1 else (0, C_F),
                    LTR
                )
            )
        else:
            bounds =  [(0, C)] * tot_qt_samples

        dual_instance = lib.Dual(H)
        alphaStar, _x, _y = sp.optimize.fmin_l_bfgs_b(
            dual_instance.l_dual,
            np.zeros(tot_qt_samples),
            bounds = bounds,
            factr = 0,
            maxiter = 10_000,
            maxfun = 100_000,
            # iprint=1
        )

        return alphaStar, _x, _y

    def _train_SVM_linear(self, DTR, LTR, DTE, LTE, C, K = 1, rebalanced:bool=False, pi_T=0.5):
        Z_LTR = lib.compute_Z(LTR = LTR)
        Z_LTE = lib.compute_Z(LTR = LTE)

        # append to DTR an array of ones (Lab09-a)
        DTR_expanded = np.vstack([DTR, np.ones((1, DTR.shape[1]))*K])

        DTE_expanded = np.vstack([DTE, np.ones((1, DTE.shape[1]))*K])

        # compute the G matrix (Lab09-b)
        G = np.dot(DTR_expanded.T, DTR_expanded)

        # (Lab09-c) 
        H = lib.colv(Z_LTR) * lib.rowv(Z_LTR) * G

        alphaStar, _x, _y = self._train_SVM(DTR = DTR, LTR = LTR, C = C, H = H, rebalanced=rebalanced, pi_T=pi_T)

        wStar = np.dot(DTR_expanded, lib.colv(alphaStar) * lib.colv(Z_LTR))

        def JPrimal(w, DT_expanded, C, Z_L):
            # Primal formulation (Lab09-e)
            S = np.dot(lib.rowv(w), DT_expanded)
            loss = np.maximum(np.zeros(S.shape), 1 - Z_L * S).sum()
            return (
                0.5 * np.linalg.norm(w)**2 + C * loss, 
                S, 
                loss
            )
        
        # # j primal with train set
        # j_primal, S, loss = JPrimal(wStar, DTR_expanded, C, Z_L=Z_LTR)
        # return j_primal, S, loss

        # dual_loss = -_x
        # dual_gap = j_primal - dual_loss
            
        # # j primal with test set
        j_primal_test, S_test, loss_test = JPrimal(wStar, DTE_expanded, C, Z_L=Z_LTE)
        return j_primal_test, S_test, loss_test

    def compute_score(self, DTE, LTE=None, C=0.1, K=1, rebalanced:bool=False, pi_T=0.5):
        """Returns scores"""
        self.C = C
        self.K = K

        _, scores, _ = self._train_SVM_linear(
            DTR=self.D,
            LTR=self.L,
            DTE=DTE,
            C=C,
            LTE=LTE,
            K=K,
            rebalanced=rebalanced,
            pi_T=pi_T,
        )
        self.scores = scores.ravel() # XXX need to ravel
        return self.scores

    def train(self, DTE, classes_prior: list):
        """Returns posteriors"""
        raise NotImplementedError

    def classify(self, LTE):
        raise NotImplementedError


class SVMKernelRBFClassifier(SVMLinearClassifier):

    def __init__(self, D, L):
        super(SVMKernelRBFClassifier, self).__init__(
            D,
            L,
        )

    def _train_SVM_kernel_RBF(self, DTR, LTR, DTE, C, LTE, K = 1, gamma = 1, rebalanced:bool=False, pi_T=0.5):
        """
        TRAINING
        """
        print(f'Training SVM RBF with C={C}, K={K}, gamma={gamma}, rebalanced ? {rebalanced}, pi_T={pi_T}')
        Z_LTR = lib.compute_Z(LTR = LTR)
        Z_LTE = lib.compute_Z(LTR = LTE)

        # cacololo di (Lab09-l)
        kernel_rbf = lambda x1, x2: np.exp( -gamma * (np.linalg.norm(x1 - x2) ** 2)) + K ** 2

        # Compute the H matrix exploiting broadcasting
        kernel_rbf_function = np.zeros((DTR.shape[1], DTR.shape[1]))
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                kernel_rbf_function[i,j] = kernel_rbf(x1=DTR[:,i], x2=DTR[:,j])
        # cacololo di (Lab09-f)
        H = lib.colv(Z_LTR) * lib.rowv(Z_LTR) * kernel_rbf_function

        alphaStar, _x, _y = self._train_SVM(DTR = DTR, LTR=LTR, C = C, H = H, rebalanced=rebalanced, pi_T=pi_T)

        """
        CLASSIFICATION
        """
        # COMPUTE SCORES FORM alphaStar and make predictions on test samples    
        # cacololo di (Lab09-g)
        S = np.zeros(DTE.shape[1]) # l'array degli scores sarà lungo come i dati di test
        # for each test sample xt(i), iterate for all training sample(j) to compute score
        for i in range(DTE.shape[1]):
            for j in range(DTR.shape[1]):
                # if alpha == 0 the point is not supervector, avoid computation
                if alphaStar[j] == 0:
                    continue
                S[i] += alphaStar[j] * Z_LTR[j] * kernel_rbf(x1 = DTR[:,j], x2 = DTE[:,i])

        return S

        accuracy, error_rate = compute_binary_accuracy_and_error(SCORE = S, LTE = Z_LTE) # Z_LTE ?
        # Compute dual loss
        dl = -_x
        print("K=%d, C=%f, RBF (gamma=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, gamma, dl, error_rate))

    def compute_score(self, DTE, LTE=None, C=0.1, K=1,  gamma=1, rebalanced:bool=False, pi_T=0.5):
        """Returns scores"""
        self.C = C
        self.K = K

        scores = self._train_SVM_kernel_RBF(
            DTR=self.D,
            LTR=self.L,
            DTE=DTE,
            C=C,
            LTE=LTE,
            K=K,
            gamma=gamma,
            rebalanced=rebalanced,
            pi_T=pi_T,
        )
        self.scores = scores.ravel() # XXX need to ravel
        return self.scores



class SVMKernelPolynomialClassifier(SVMLinearClassifier):

    def __init__(self, D, L):
        super(SVMKernelPolynomialClassifier, self).__init__(
            D,
            L,
        )

    def train_SVM_kernel_Polynomial(self, DTR, LTR, DTE, C, LTE, K = 1, _c = 1, rebalanced:bool=False, pi_T=0.5):
        """
        TRAINING
        """
        polynomial_degree = 2
        d = polynomial_degree
        print(f'Training SVM polynomial with C={C}, K={K}, _c={_c}, polynomial degree={polynomial_degree}, rebalanced ? {rebalanced}, pi_T={pi_T}')
        Z_LTR = lib.compute_Z(LTR = LTR)
        Z_LTE = lib.compute_Z(LTR = LTE)


        # cacololo di (Lab09-h)
        polynomial_rbf = lambda x1, x2: (np.dot(x1.T, x2) + _c) ** d + K ** 2

        # Compute the H matrix exploiting broadcasting
        kernel_poly_funnction = np.zeros((DTR.shape[1], DTR.shape[1]))
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                kernel_poly_funnction[i,j] = polynomial_rbf(x1=DTR[:,i], x2=DTR[:,j])
        # cacololo di (Lab09-f)
        H = lib.colv(Z_LTR) * lib.rowv(Z_LTR) * kernel_poly_funnction

        alphaStar, _x, _y = self._train_SVM(DTR = DTR, LTR=LTR, C = C, H = H, rebalanced=rebalanced, pi_T=pi_T)

        """
        CLASSIFICATION
        """
        # COMPUTE SCORES FORM alphaStar and make predictions on test samples    
        # cacololo di (Lab09-g)
        S = np.zeros(DTE.shape[1]) # l'array degli scores sarà lungo come i dati di test
        # for each test sample xt(i), iterate for all training sample(j) to compute score
        for i in range(DTE.shape[1]):
            for j in range(DTR.shape[1]):
                # if alpha == 0 the point is not supervector, avoid computation
                if alphaStar[j] == 0:
                    continue
                S[i] += alphaStar[j] * Z_LTR[j] * polynomial_rbf(x1 = DTR[:,j], x2 = DTE[:,i])

        return S

        accuracy, error_rate = compute_binary_accuracy_and_error(SCORE = S, LTE = Z_LTE)
        # Compute dual loss
        dl = -_x

        print("K=%d, C=%f, Kernel Poly (d=%d, _c=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, d, _c, dl, error_rate))

    def compute_score(self, DTE, LTE=None, C=0.1, K=1, _c=1, rebalanced:bool=False, pi_T=0.5):
        """Returns scores"""
        self.C = C
        self.K = K

        scores = self.train_SVM_kernel_Polynomial(
            DTR=self.D,
            LTR=self.L,
            DTE=DTE,
            C=C,
            LTE=LTE,
            K=K,
            _c=_c,
            rebalanced=rebalanced,
            pi_T=pi_T,
        )
        self.scores = scores.ravel() # XXX need to ravel
        return self.scores


class GMMClassifierMixin:
    def GMM_scores_per_sample(self, X, gmm):
        """
        For each sample in X compute 

        :X data of training
        :gmm json like array where each element is a Cluster
            and each element contains
            [
                [
                    {
                        "weight": weight1,
                        "mean": [[mu1]],         // Cluster 1
                        "cov": [[Cov1]]
                    }
                ],
                ...
            ]
        """
        G = len(gmm) # nr of clusters
        N = X.shape[1] # nr of samples
        S = np.zeros((G, N)) # Scores matrix G x N
        # loop for each component
        for g in range(G):
            mu = gmm[g]['mean'] # array of mean
            weight = np.log(gmm[g]['weight']) # (prior)
            Cov = gmm[g]['cov']
            # fill Scores matrix line by line (each line a cluster)
            S[g, :] = lib.logpdf_GAU_ND(X=X, mu=mu, C=Cov)

        return S

    def GMM_ll_per_sample(self, scores, gmm):
        G = len(gmm) # nr of clusters

        for g in range(G):
            weight = np.log(gmm[g]['weight']) # (prior)
            # add prior (weight) to scores
            scores[g, :] += weight
        
        # sum along Y axis (lab10_b)
        return sp.special.logsumexp(scores, axis=0)

    # PROF FUNCTION
    def GMM_EM(
        self,
        X, 
        gmm, 
        stop_threshold = 1e-6, 
        constraint=False, 
        constraint_threshold:float=0.01, 
        diagonal:bool=False, 
        tied:bool=False
    ):
        """
        Computes the EM alghoritm steps
        :X             - samples
        :gmm           - the gmm values for each cluster [(weight, [[mean]], [[Cov]])]
        :stop_threshold - if log likehood doesn't increase at least by the threshold value in the
                        next iteration, stop the algorithm
        :constraint:   - if True constraints the eigenvalues to be larger than the 
                        :constraint_threshold
                        to avoid degenerative solution due to shrink of Covariance around 
                        one single sample
        :diagonal      - if True at each step the non-diagonal elements of Cov will be zeroed
        :tied          - if True the gmm's Cov matrix will be the same and will be computed as
                        the mean between Components Cov Matrices weighted on the nr of samples 
        """

        # log likehood values at previous and current iteration
        ll_old = None
        ll_new = None

        G = len(gmm)
        N = X.shape[1]

        times_llr_decrease = 0
        while (ll_old is None) or (ll_new - ll_old > stop_threshold):
            ll_old = ll_new
            ### E -Step
            SJ = np.zeros((G, N))
            for g in range(G):
                mu = gmm[g]['mean']
                weight = np.log(gmm[g]['weight']) # (prior)
                Cov = gmm[g]['cov']
                SJ[g, :] = lib.logpdf_GAU_ND(X=X, mu=mu, C=Cov)
                # add prior (weight) to scores
                SJ[g, :] += weight
            # sum along Y axis (lab10_b)
            SM = sp.special.logsumexp(SJ, axis=0)
            # sum of all N log-likehood (it will be the likehood of 
            # the whole samples since we suppose them to be independent)
            # weighted for the number of sample (lab10_c)
            ll_new = SM.sum()/N 
            # posterior (in log domain is Joint - Marginal) - thena again to exp
            # lab10_d
            Posterior = np.exp(SJ - SM)

            ### M -Step
            # compute new set of parameters for GMM
            gmm_new = []
            # for each Cluster
            for g in range(G):
                # row by row of Posterior Matrix
                gamma = Posterior[g, :]
                # compute Statistics lab10_g
                Z = gamma.sum() # nr of sample associated with cluster g
                F = (lib.rowv(gamma) * X).sum(1)
                S = np.dot(X, (lib.rowv(gamma) * X).T)
                # compute the new weight, mean, Cov from statistics
                w = Z / N # the sum of Z order is exactly the nr of samples
                mu = lib.colv(F / Z)
                Sigma = S / Z - np.dot(mu, mu.T)
                if diagonal:
                    # Baesyan Gaussian Model
                    Sigma = Sigma * np.eye(Sigma.shape[0]) # put all 0s in non-diagonal elements
                gmm_new.append(
                    {
                        "weight": w, 
                        "mean": mu, 
                        "cov": Sigma,
                        "Z": Z
                    }
                )

            # for tied I have to average the Cov of the
            #new computed GMMs so I have to reiterate
            if tied:
                Tied_Cov = np.zeros(Sigma.shape)
                # compute tied cov lab10_h
                for g in range(G):
                    Tied_Cov +=  gmm_new[g]['Z'] * gmm_new[g]['cov']
                Tied_Cov = Tied_Cov/X.shape[1]
                # if we are constriting the Cov, recompute it (see docs)
                # constraints must be evaluated after Tied or Diagonal
                # computation (see Lab10)
                if constraint:
                    Tied_Cov = self.compute_Cov_constrained(Cov=Tied_Cov, constraint_threshold=constraint_threshold)
                # update GMMs with new tied Cov
                for g in range(G):
                    gmm_new[g]['cov'] = Tied_Cov
            else:
                if constraint:
                    for g in range(G):
                        gmm_new[g]['cov'] = self.compute_Cov_constrained(Cov=gmm_new[g]['cov'], constraint_threshold=constraint_threshold)

            gmm = gmm_new
            
            ## llr should always increase
            if ll_new is not None and ll_old is not None:
                if ll_new-ll_old < 0:
                    times_llr_decrease =+ 1
                    if ll_new-ll_old < 0 and abs(ll_new-ll_old) > 10**(-3):
                        print('old_ll', ll_old)
                        print('new_ll', ll_new)
                        print('/warning LLR should increase at each iteration')
                        # raise ValidationErr('LLR should increase at each iteration')
                    if times_llr_decrease > 3:
                        print('LLR decreased for 3 times')

                        raise ValidationErr('LLR decreased for 3 times')
        return gmm

    def compute_Cov_constrained(self, Cov, constraint_threshold:float=0.01):
        """
        lab10_t
        constraints the eigenvalues to be larger than the 
        :constraint_threshold
        to avoid degenerative solution due to shrink of Covariance around 
        one single sample

        Return:
            Cov constrained
        """
        # print('Cov before constraint', Cov)
        U, s, _ = np.linalg.svd(Cov)
        # force all the eighen-vallues lower thant the constraint_threshold
        # to be at least = to the constraint_threshold
        s[s < constraint_threshold] = constraint_threshold
        Cov = np.dot(U, lib.colv(s) * U.T)
        # print('Cov after constraint', Cov)
        return Cov

    def _LBG_split(self, gmm, alpha = 0.1):
        """
        Split samples as from LBG algorithm
        returns the given GMM splitted in a 2x
        """
        gmm_splitted = []
        # loop for each of the GMM and split them
        for i in range(len(gmm)):
            weight = gmm[i]['weight']
            mu = gmm[i]['mean']
            Cov = gmm[i]['cov']
            # obtain the sorted eighen values and vectors
            # from Cov matrix lab10_f
            U, s, Vh = np.linalg.svd(Cov)
            displacement = U[:, 0:1] * s[0] ** 0.5 * alpha

            gmm_splitted.append(
                {
                    "weight": weight / 2, 
                    "mean": mu + displacement, 
                    "cov": Cov
                }
            )
            gmm_splitted.append(
                {
                    "weight": weight / 2, 
                    "mean": mu - displacement, 
                    "cov": Cov
                }
            )

        return gmm_splitted

    def _GMM_LBG_algorithm(
        self,
        dataset, 
        gmm, 
        nr_iterations, 
        first_call:bool=True, 
        constraint:bool=False, 
        constraint_threshold:float=0.01,
        diagonal:bool=False,
        tied:bool=False
    ):
        """
        Computes the EM algorithm using LBG to calculate first gmm values
        :dataset       - samples
        :gmm           - the gmm values for each cluster [{"weight": weight, "mean": [[mean]], "cov": [[Cov]]}]
        :first_call    - must be True only the first time it is called, to understand
                        that the :gmm param is the first set of parameters for which we should
                        also compute the constraint if :constraint is True.
        :nr_iterations - how many split (e.g. if 2 we'll receive 4 cluster, 
                        if 4 we'll receive 16 clusters...)
        :constraint:   - if True constraints the eigenvalues to be larger than the 
                        :constraint_threshold
                        to avoid degenerative solution due to shrink of Covariance around 
                        one single sample
        :diagonal      - if True at each step the non-diagonal elements of Cov will be zeroed
        :tied          - if True the gmm's Cov matrix will be the same and will be computed as
                        the mean between Components Cov Matrices weighted on the nr of samples 
        """
        if nr_iterations:
            if first_call and constraint:
                Cov = gmm[0]['cov']
                Cov = self.compute_Cov_constrained(Cov=Cov, constraint_threshold=constraint_threshold)
                gmm[0]['cov'] = Cov
            first_GMM = self.GMM_EM(
                X=dataset, 
                gmm=gmm, 
                constraint=constraint, 
                constraint_threshold=constraint_threshold,
                diagonal=diagonal,
                tied=tied
            )

            GMM_splitted = self._LBG_split(first_GMM)
            # recursion
            return self._GMM_LBG_algorithm(
                dataset=dataset, 
                gmm=GMM_splitted, 
                nr_iterations=nr_iterations-1, 
                first_call=False,
                constraint=constraint,
                constraint_threshold=constraint_threshold,
                diagonal=diagonal,
                tied=tied
            )
        gmm = self.GMM_EM(
                X=dataset, 
                gmm=gmm, 
                constraint=constraint, 
                constraint_threshold=constraint_threshold,
                diagonal=diagonal,
                tied=tied
        )
        return gmm

    def _gmm_scores(
        self,
        DTR, 
        LTR, 
        DTE, 
        LTE, 
        algorithm:Literal['full_cov', 'diag_cov', 'tied_cov'],
        nr_clusters,
        constraint:bool=True,
        constraint_threshold:float=0.01
    ):
        # split DTR in different classes based on their labels
        data_splitted_by_class_list = [] # array with data splitted for classes
        for label in set(LTR):
            data_splitted_by_class_list.append(
                DTR[:, LTR == label]
            )

        # set algorithm type
        tied = False
        diagonal = False
        if algorithm == 'diag_cov':
            diagonal = True
        elif algorithm == 'tied_cov':
            tied = True
        else:
            tied = False
            diagonal = False
        
        # compute the likehood for each Training Class
        marginal_likehood = []
        scores_list = []
        for data in data_splitted_by_class_list:
            mean, cov = lib.compute_empirical_mean_and_cov(
                X = data, # all sample of one klass
                is_bayesian = diagonal
            )

            initial_GMM = [
                {
                    "weight": 1.0, 
                    "mean": mean, 
                    "cov": cov
                }
            ]

            result_GMM = self._GMM_LBG_algorithm(
                dataset=data, 
                gmm=initial_GMM, 
                nr_iterations= np.floor(np.log2(nr_clusters)), # log_2 because each LBG iteration slit the gmm in 2
                constraint=constraint,
                constraint_threshold=constraint_threshold,
                diagonal=diagonal,
                tied=tied
            )

            test_data_scores = self.GMM_scores_per_sample(
                X= DTE,
                gmm=result_GMM,
            )
            
            scores_list.append(test_data_scores)

            # for each GMM compute the Posterior of Test Data
            test_data_posterior = self.GMM_ll_per_sample(
                scores = test_data_scores,
                gmm = result_GMM
            )


            marginal_likehood.append(test_data_posterior)

        scores_matrix_for_cluster = np.vstack(scores_list)
        # return scores_matrix_for_cluster


        # Matrix row=Clusters, columns=Posteriors
        Posteriors_Matrix_for_cluster = np.vstack(marginal_likehood)

        return Posteriors_Matrix_for_cluster[1] - Posteriors_Matrix_for_cluster[0]

        # # Compute the predicted labels
        # predicted_lab = np.argmax(Posteriors_Matrix_for_cluster, axis=0)
        # nr_correct_predictions = np.array(predicted_lab == LTE).sum()
        # accuracy = nr_correct_predictions/LTE.size*100
        # error_rate = 100-accuracy
        # return error_rate


class GmmClassifier(BaseClassifier, GMMClassifierMixin):
    """
    >>> score
    >>> train
    >>> classify

    """
    def __init__(self, D, L):
        super().__init__(
            D,
            L
        )
        self.scores = None



    def compute_score(
        self, 
        DTE, 
        LTE=None, 
        stop_threshold=1e-6, 
        constraint_threshold=0.01, 
        alpha=0.1,
        algorithm:Literal['full_cov', 'diag_cov', 'tied_cov']='full_cov',
        nr_clusters=0,
    ):
        """
        :DTE
        :LTE
        :stop_threshold (delta_t)   = stop criteria for the EM algorithm
        :constraint_threshold (psi) = constraint on eigenvalues not to shrink cov matrix
        :alpha                      = offset in the LBG split
        """
        print(f'GMM computing scores with type={algorithm}, nr_cluster={nr_clusters}')
        self.scores = self._gmm_scores(
            DTR=self.D,
            LTR=self.L,
            LTE=LTE,
            DTE=DTE,
            algorithm=algorithm,
            nr_clusters=nr_clusters,
            constraint_threshold=constraint_threshold,
        )
        return self.scores


    def train(self, classes_prior: list):
        raise NotImplementedError

    def classify(self, LTE):
        raise NotImplementedError
