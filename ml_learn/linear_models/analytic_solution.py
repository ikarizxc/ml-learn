from ml_learn.linear_models.linear_base import LinearBase
from ml_learn.utils.regression_metrics import mean_squared_error
import numpy as np

class AnalyticLinearRegression(LinearBase):
    def __init__(self, reg_strength=0.0, fit_intercept=True, verbose=False):
        super().__init__(reg_strength, fit_intercept, verbose)
        self.loss_func = mean_squared_error

    def _update_weights(self, X, y):
        S = X.T @ X
        self.w = np.linalg.pinv(S + self.reg_strength * np.eye(S.shape[1])) @ X.T @ y

        if self.verbose:
            print(f"Loss: {self._compute_loss(y, X.dot(self.w))}")

class AnalyticLinearClassifier(AnalyticLinearRegression):
    def __init__(self, reg_strength=0.0, fit_intercept=True, verbose=False):
        super().__init__(reg_strength, fit_intercept, verbose)

    def predict(self, X):
        pred = super().predict(X)
        return (pred >= 0).astype(int)