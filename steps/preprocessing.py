from sklearn.externals import joblib
from sklearn.feature_extraction import text

from .base import BaseTransformer


class FillNA(BaseTransformer):
    def __init__(self, na_columns):
        self.na_columns = na_columns

    def transform(self, X):
        X[self.na_columns] = X[self.na_columns].fillna("unknown").values
        return {'X': X}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.na_columns = params['na_columns']
        return self

    def save(self, filepath):
        params = {'na_columns': self.na_columns,
                  }
        joblib.dump(params, filepath)


class XYSplit(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, meta_valid=None, train_mode=True):
        X = meta[self.x_columns].values
        if train_mode:
            y = meta[self.y_columns].values
        else:
            y = None

        if meta_valid is not None:
            X_valid = meta_valid[self.x_columns].values
            y_valid = meta_valid[self.y_columns].values
            valid = X_valid, y_valid
        else:
            valid = None

        return {'X': X,
                'y': y,
                'validation_data': valid,
                'train_mode': train_mode}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class TfidfVectorizer(BaseTransformer):
    def __init__(self, **kwargs):
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def fit(self, text):
        self.vectorizer.fit(text)
        return self

    def transform(self, text):
        return {'features': self.vectorizer.transform(text)}

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)
