import numpy as np

def calc_tp(y_true, y_pred):
    return np.count_nonzero((y_true == 1) & (y_pred == 1))

def calc_tn(y_true, y_pred):
    return np.count_nonzero((y_true == 0) & (y_pred == 0))

def calc_fp(y_true, y_pred):
    return np.count_nonzero((y_true == 0) & (y_pred == 1))

def calc_fn(y_true, y_pred):
    return np.count_nonzero((y_true == 1) & (y_pred == 0))

def confusion_matrix(y_true, y_pred):
    return [[calc_tp(y_true, y_pred), calc_fp(y_true, y_pred)],
            [calc_fn(y_true, y_pred), calc_tn(y_true, y_pred)]]
    
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    return calc_tp(y_true, y_pred) / (calc_tp(y_true, y_pred) + calc_fp(y_true, y_pred))

def recall(y_true, y_pred):
    return calc_tp(y_true, y_pred) / (calc_tp(y_true, y_pred) + calc_fn(y_true, y_pred))

def f1_score(y_true, y_pred):
    return 2 * (precision(y_true, y_pred) * recall(y_true, y_pred)) / (precision(y_true, y_pred) + recall(y_true, y_pred))

def f_beta_score(y_true, y_pred, beta):
    return (1 + beta ** 2) * (precision(y_true, y_pred) * recall(y_true, y_pred)) / (beta ** 2 * precision(y_true, y_pred) + recall(y_true, y_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(y_true, y_pred):
    p = sigmoid(y_pred)
    
    eps = 1e-15
    p = np.clip(p, eps, p - eps)
    return np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1-p))