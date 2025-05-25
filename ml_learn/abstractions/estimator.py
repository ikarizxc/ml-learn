from ml_learn.utils.plot import plot_to_base64
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Estimator(ABC):
    """
    Base class for models
    """
    def __init__(self, verbose):
        self.verbose = verbose
        self.loss_func = None
        self.loss_history = []

    @abstractmethod
    def fit(self, X, y):
        """
        Train model on data X and y
        
        X : numpy array of shape (n_samples, n_feaures)
        y : numpy array of shape (n_samples)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        Predict values/labels for data X
        
        X : numpy array of shape (n_samples, n_features)
        
        Return: 
            numpy array of shape (n_samples)
        """
        pass

    def _compute_loss(self, y_true, y_pred):
        """
        Calculate loss with setted loss function
        
        y_true : numpy array of shape (n_samples)
        y_pred : numpy array of shape (n_samples)
        
        Return: float
        """
        return self.loss_func(y_true, y_pred)

    def set_loss_func(self, loss_func):
        """
        Set loss function for model
        
        loss_func : function which takes 2 numpy array of shape (n_samples) and gives float 
        
        Return: float
        """
        self.loss_func = loss_func
        
    def get_metrics(self):
        metrics = {}
        
        fig = plt.figure()
        plt.plot(self.loss_history, label='Loss')
        plt.title('Loss curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        metrics['loss'] = plot_to_base64(fig)
        
        return metrics