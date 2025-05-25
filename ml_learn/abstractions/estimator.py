from abc import ABC, abstractmethod

class Estimator(ABC):
    def __init__(self, verbose):
        self.verbose = verbose
        self._loss_func = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _compute_loss(self, y_pred, y_true):
        return self._loss_func(y_pred, y_true)

    def set_loss_func(self, loss_func):
        self._loss_func = loss_func