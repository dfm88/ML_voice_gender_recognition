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
    def score(self, DTE):
        """Returns posteriors"""
        raise NotImplementedError

    @abstractmethod
    def train(self, DTE, classes_prior: list):
        """Returns posteriors"""
        raise NotImplementedError

    @abstractmethod
    def classify(self, LTE):
        raise NotImplementedError


class GaussianClassifier(BaseClassifier):

    def __init__(self, D, L):
        super(GaussianClassifier, self).__init__(
            D,
            L
        )
        self.mean = lib.compute_empirical_mean(self.D)
        self.cov  = lib.compute_empirical_cov(self.D)
        self.posteriors = None

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
        import ipdb; ipdb.set_trace()
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


    def score(self, DTE):
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
                D_train_for_class
            )
        self.scores = self._log_domain_scores(
            class_gaussian_param=class_gaussian_parameters,
            class_labels=class_labels,
            DTE=DTE,
        )



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
        import ipdb; ipdb.set_trace()
        predicted_lab = np.argmax(self.posteriors, axis=0)
        nr_correct_predictions = np.array(predicted_lab == LTE).sum()
        accuracy = nr_correct_predictions/LTE.size*100
        error_rate = 100-accuracy
        return error_rate


