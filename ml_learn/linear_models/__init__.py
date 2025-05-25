from .linear_base import LinearBase
from .analytic_solution import AnalyticLinearClassifier, AnalyticLinearRegression
from .gradient_descent import GradientDescentRegression, LogisticRegression

__all__ = ['AnalyticLinearClassifier',
           'AnalyticLinearRegression',
           'GradientDescentRegression',
           'LogisticRegression']