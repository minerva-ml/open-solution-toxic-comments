import numpy as np
from scipy.optimize import minimize
from sklearn.externals import joblib
from tqdm import tqdm

from steps.base import BaseTransformer


class Blender(BaseTransformer):
    def __init__(self, func, min, method, runs, maxiter):
        self.func = func
        self.min = min
        self.method = method
        self.runs = runs
        self.maxiter = maxiter

    def _optim_func(self, X, y):
        def f(weights):
            weights = weights.reshape(1, self.nr_models)
            weighted_predictions = np.sum(X * weights, axis=-1)
            if self.min:
                return -1.0 * self.func(y, weighted_predictions)
            else:
                return self.func(y, weighted_predictions)
        return f

    def fit(self, X, y):
        self.nr_models = X.shape[-1]
        res_list = []
        for _ in tqdm(range(self.runs)):
            starting_values = np.random.uniform(size=(1, self.nr_models))

            bounds = [(0, 1)] * self.nr_models

            res = minimize(self._optim_func(X, y),
                           starting_values,
                           method=self.method,
                           bounds=bounds,
                           options={'disp': False,
                                    'maxiter': self.maxiter})
            res_list.append(res)
        self.best_run = sorted(res_list, key=lambda x: x['fun'])[0]
        self.best_weights = self.best_run['x'].reshape(1, self.nr_models) / self.nr_models
        return self

    def transform(self, X, y=None):
        predictions = np.sum(X * self.best_weights, axis=-1)
        return {'predictions': predictions}

    def load(self, filepath):
        obj = joblib.load(filepath)
        self.best_weights = obj['best_weights']
        self.best_run = obj['best_run']
        return self

    def save(self, filepath):
        joblib.dump({'best_weights': self.best_weights,
                     'best_run': self.best_run}, filepath)


class Clipper(BaseTransformer):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def transform(self, predictions):
        if self.lower is not None:
            predictions = np.where(predictions < self.lower, 0, predictions)
        if self.upper is not None:
            predictions = np.where(predictions > self.upper, 1, predictions)
        return {'predictions': predictions}

    def load(self, filepath):
        obj = joblib.load(filepath)
        self.lower = obj['lower']
        self.upper = obj['upper']
        return self

    def save(self, filepath):
        joblib.dump({'lower': self.lower,
                     'upper': self.upper}, filepath)
