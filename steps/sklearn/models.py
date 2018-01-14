import numpy as np
import sklearn.linear_model as lr
from sklearn import svm
from sklearn.ensamble import RandomForestClassifier
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

from utils import multi_log_loss

logger = get_logger()


class MultilabelEstimator(BaseTransformer):
    @property
    def estimator(self):
        return NotImplementedError
    
    def __init__(self, label_nr, **kwargs):
        self.label_nr = label_nr
        self.estimators = self._get_estimators(**kwargs)

    def _get_estimators(self, **kwargs):
        estimators = []
        for i in range(self.label_nr):
            estimators.append((i, self.estimator(**kwargs)))
        return estimators

    def fit(self, X, y):
        for i, estimator in self.estimators:
            logger.info('fitting estimator {}'.format(i))
            estimator.fit(X, y[:, i])
        return self

    def transform(self, X, y=None):
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
        self.estimator = params['estimator']
        return self

    def save(self, filepath):
        params = {'label_nr': self.label_nr,
                  'estimator': self.estimator}
        joblib.dump(params, filepath)
        

class LogisticRegressionMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return lr.LogisticRegression
    
    
class SVCMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return lr.svm.SVC

    
class RandomForestMultilabel(BaseTransformer):
    @property
    def estimator(self):
        return RandomForestClassifier