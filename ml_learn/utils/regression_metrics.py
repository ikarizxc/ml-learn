import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean(np.pow(y_true - y_pred, 2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.pow(y_true - y_pred, 2)))

def r2(y_true, y_pred):
    return 1 - (np.sum(np.pow(y_true - y_pred, 2)) / np.sum(np.pow(y_true - np.mean(y_true), 2)))