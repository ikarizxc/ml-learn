from ml_learn.linear_models.linear_base import LinearBase
from ml_learn.utils.classification_metrics import log_loss, sigmoid
from ml_learn.utils.regression_metrics import mean_squared_error
from abc import abstractmethod
import numpy as np

class GradientDescentBase(LinearBase):
    def __init__(self, learning_rate=1e-2, reg='l2', epochs=1000, epsilon=1e-4, reg_strength=1e-2, batch_size=None, n_epochs_to_stop=5, shuffle=False, fit_intercept=True, verbose=False):
        super().__init__(reg_strength, fit_intercept, verbose)
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_epochs_to_stop = n_epochs_to_stop
        self.shuffle = shuffle
        self.loss_func = mean_squared_error

    def _update_weights(self, X, y):
        n, m = X.shape
        self.w = np.random.randn(m) * 0.01
        self.batch_size = n if self.batch_size is None else self.batch_size
        
        best_loss = float('inf')
        n_epochs_without_changes = 0
        
        for epoch in range(self.epochs):            
            if self.shuffle:
                shuffled_indicies = np.random.permutation(n)
                X = X[shuffled_indicies]
                y = y[shuffled_indicies]

            for batch_start in range(0, n, self.batch_size):
                X_batch = X[batch_start:batch_start+self.batch_size]
                y_batch = y[batch_start:batch_start+self.batch_size]
                w_new = self.w - self.learning_rate * self._calculate_gradient(X_batch, y_batch)
                self.w = w_new

            cur_loss = self._compute_loss(y, X.dot(self.w))
            
            self.loss_history.append(cur_loss)
            
            if self.verbose:
                print(f"Epoch {epoch}. Loss: {cur_loss}")
            
            if cur_loss < best_loss - self.epsilon:
                best_loss = cur_loss
                n_epochs_without_changes = 0
            else:
                n_epochs_without_changes += 1
                
            if n_epochs_without_changes >= self.n_epochs_to_stop:
                if self.verbose:
                    print(f"Stopped on epoch {epoch}")
                break
    
    @abstractmethod
    def _calculate_gradient(self, X, y):
        """
        Calculate gradient with regularization
        
        X : numpy array of shape (n_samples, n_feaures)
        y : numpy array of shape (n_samples)
        
        Return:
            numpy array of shape (n_samples)
        """
        pass

class GradientDescentRegression(GradientDescentBase):
    def __init__(self, learning_rate=1e-2, reg='l2', epochs=1000, epsilon=1e-4, reg_strength=1e-2, n_epochs_to_stop=5, batch_size=None, shuffle=True, fit_intercept=True, verbose=False):
        super().__init__(learning_rate=learning_rate, reg=reg, epochs=epochs, epsilon=epsilon, reg_strength=reg_strength, batch_size=batch_size, fit_intercept=fit_intercept, verbose=verbose, shuffle=shuffle, n_epochs_to_stop=n_epochs_to_stop)
        
    def _calculate_gradient(self, X, y):
        n, m = X.shape

        grad = 2 / n * X.T.dot(X.dot(self.w) - y)

        if self.reg == 'l1':
            grad += self.reg_strength * np.sign(self.w)
        elif self.reg == 'l2':
            grad += 2 * self.reg_strength * self.w
        elif self.reg is not None:
            raise Exception("'reg' parameter value must be 'l1', 'l2' or None")

        return grad
  