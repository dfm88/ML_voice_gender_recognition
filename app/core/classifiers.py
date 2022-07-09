from abc import ABC, abstractmethod, abstractclassmethod
import sys
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
                # import ipdb; ipdb.set_trace()
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
            empirical_prior_T = DTR[:LTR==1].shape[1] / tot_qt_samples
            empirical_prior_F = DTR[:LTR==0].shape[1] / tot_qt_samples
        
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
            maxiter = 100_000,
            maxfun = 100_000,
            # iprint=1
        )

        return alphaStar, _x, _y

    # def _train_SVM_linear(self, DTR, LTR, DTE, LTE, C, K = 1):
    #     Z_LTR = lib.compute_Z(LTR = LTR)
    #     Z_LTE = lib.compute_Z(LTR = LTE)

    #     # append to DTR an array of ones (Lab09-a)
    #     DTR_expanded = np.vstack([DTR, np.ones((1, DTR.shape[1]))*K])

    #     DTE_expanded = np.vstack([DTE, np.ones((1, DTE.shape[1]))*K])

    #     # compute the G matrix (Lab09-b)
    #     G = np.dot(DTR_expanded.T, DTR_expanded)

    #     # H = lib.colv(Z_LTR) * lib.rowv(Z_LTR) * G
    #     H = lib.colv(LTR) * lib.rowv(LTR) * G
    #     alphaStar, _x, _y = self._train_SVM(DTR = DTR, C = C, H = H)

    #     # wStar = np.dot(DTR_expanded, lib.colv(alphaStar) * lib.colv(Z_LTR))
    #     wStar = np.dot(DTR_expanded, lib.colv(alphaStar) * lib.colv(LTR))

    #     def JPrimal(w, DT_expanded, C, Z_L):
    #         # Primal formulation (Lab09-e)
    #         S = np.dot(lib.rowv(w), DT_expanded)
    #         loss = np.maximum(np.zeros(S.shape), 1 - Z_L * S).sum()
    #         return (
    #             0.5 * np.linalg.norm(w)**2 + C * loss, 
    #             S, 
    #             loss
    #         )
        
    #     # # j primal with train set
    #     # j_primal, S, loss = JPrimal(wStar, DTR_expanded, C, Z_L=Z_LTR)
    #     # return j_primal, S, loss

    #     # dual_loss = -_x
    #     # dual_gap = j_primal - dual_loss
            
    #     # j primal with test set
    #     j_primal_test, S_test, loss_test = JPrimal(wStar, DTE_expanded, C, Z_L=Z_LTE)
    #     return j_primal_test, S_test, loss_test

    def _train_SVM_linear(self, DTR, LTR, DTE, LTE, C, K = 1):
        Z_LTR = lib.compute_Z(LTR = LTR)
        Z_LTE = lib.compute_Z(LTR = LTE)

        # append to DTR an array of ones (Lab09-a)
        DTR_expanded = np.vstack([DTR, np.ones((1, DTR.shape[1]))*K])

        DTE_expanded = np.vstack([DTE, np.ones((1, DTE.shape[1]))*K])

        # compute the G matrix (Lab09-b)
        G = np.dot(DTR_expanded.T, DTR_expanded)

        # (Lab09-c) 
        H = lib.colv(Z_LTR) * lib.rowv(Z_LTR) * G

        alphaStar, _x, _y = self._train_SVM(DTR = DTR, LTR = LTR, C = C, H = H)

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

    def compute_score(self, DTE, LTE=None, C=0.1, K=1):
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
        )
        self.scores = scores.ravel() # XXX need to ravel
        return self.scores

    def train(self, DTE, classes_prior: list):
        """Returns posteriors"""
        raise NotImplementedError

    def classify(self, LTE):
        raise NotImplementedError