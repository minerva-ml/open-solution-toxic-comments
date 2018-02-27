import numpy as np
import sklearn.linear_model as lr
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from catboost import CatBoostClassifier

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class MultilabelEstimator(BaseTransformer):
    def __init__(self, label_nr, **kwargs):
        self.label_nr = label_nr
        self.estimators = self._get_estimators(**kwargs)

    @property
    def estimator(self):
        return NotImplementedError

    def _get_estimators(self, **kwargs):
        estimators = []
        for i in range(self.label_nr):
            estimators.append((i, self.estimator(**kwargs)))
        return estimators

    def fit(self, X, y, **kwargs):
        for i, estimator in self.estimators:
            logger.info('fitting estimator {}'.format(i))
            estimator.fit(X, y[:, i])
        return self

    def transform(self, X, y=None, **kwargs):
        predictions = []
        for i, estimator in self.estimators:
            prediction = estimator.predict_proba(X)
            predictions.append(prediction)
        predictions = np.stack(predictions, axis=0)
        predictions = predictions[:, :, 1].transpose()
        return {'prediction_probability': predictions}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.label_nr = params['label_nr']
        self.estimators = params['estimators']
        return self

    def save(self, filepath):
        params = {'label_nr': self.label_nr,
                  'estimators': self.estimators}
        joblib.dump(params, filepath)


class LogisticRegressionMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return lr.LogisticRegression


class SVCMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return svm.SVC


class LinearSVC_proba(svm.LinearSVC):
    def __platt_func(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions).reshape(-1, 1)
        prob_positive = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        prob_negative = 1.0 - prob_positive
        probs = np.hstack([prob_negative, prob_positive])
        print(prob_positive)
        return probs


class LinearSVCMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return LinearSVC_proba


class RandomForestMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return RandomForestClassifier


class CatboostClassifierMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return CatBoostClassifier
