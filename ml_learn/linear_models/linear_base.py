from abc import abstractmethod
import numpy as np

from ml_learn.abstractions.estimator import Estimator

class LinearBase(Estimator):
    def __init__(self, reg_strength=0.0, fit_intercept=True, verbose=False):
        super().__init__(verbose)
        self.reg_strength = reg_strength
        self.fit_intercept = fit_intercept
        self.w = None

    @abstractmethod
    def _update_weights(self, X, y):
        """
        Calculate weights 
        
        X : numpy array of shape (n_samples, n_feaures)
        y : numpy array of shape (n_samples)
        """
        pass

    def fit(self, X, y):
        Xb = self._add_intercept(X)
        self._update_weights(Xb, y)

    def _add_intercept(self, X):
        """
        Add intecept in data X if needed
        
        X : numpy array of shape (n_samples, n_feaures)
        
        Return:
            numpy array of shape (n_samples, n_features + 1)
        """
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
        if self.verbose:
            print('Added intercept')
        return X
    
    def predict(self, X):
        Xb = self._add_intercept(X)
        if self.w is None:
            print('Model must be fitted')
        return Xb.dot(self.w)