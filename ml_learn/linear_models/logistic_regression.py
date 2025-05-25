from ml_learn.utils.classification_metrics import log_loss, sigmoid      
from ml_learn.linear_models.gradient_descent import GradientDescentBase
import numpy as np

class LogisticRegression(GradientDescentBase):
    def __init__(self, learning_rate=1e-2, reg='l2', epochs=1000, epsilon=1e-3, reg_strength=1e-2, n_epochs_to_stop=5, threshold=0.5, batch_size=None, shuffle=False, fit_intercept=True, verbose=False):
        super().__init__(learning_rate=learning_rate, reg=reg, epochs=epochs, epsilon=epsilon, reg_strength=reg_strength, batch_size=batch_size, fit_intercept=fit_intercept, verbose=verbose, shuffle=shuffle, n_epochs_to_stop=n_epochs_to_stop)
        self.threshold = threshold
        self.loss_func = log_loss
        
    def _calculate_gradient(self, X, y):
        n, m = X.shape

        grad = - 1 / n * X.T.dot(y - sigmoid(X.dot(self.w)))

        if self.reg == 'l1':
            grad += self.reg_strength * np.sign(self.w)
        elif self.reg == 'l2':
            grad += 2 * self.reg_strength * self.w
        elif self.reg is not None:
            raise Exception("'reg' parameter value must be 'l1', 'l2' or None")

        return grad
        
    def predict_proba(self, X):
        """
        Predict probabilities
        
        X : numpy array of shape (n_samples, n_feaures)
        
        Return:
            numpy array of shape (n_samples)
        """
        pred = super().predict(X)
        return sigmoid(pred)
        
    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)