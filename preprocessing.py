import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from steps.base import BaseTransformer
from utils import get_logger

logger = get_logger()


def split_train_data(data_dir, validation_size):
    meta_train_filepath = os.path.join(data_dir, 'train.csv')
    meta_train_split_filepath = meta_train_filepath.replace('train', 'train_split')
    meta_valid_split_filepath = meta_train_filepath.replace('train', 'valid_split')

    logger.info('reading data from {}'.format(meta_train_filepath))
    meta_data = pd.read_csv(meta_train_filepath)
    logger.info('splitting data')
    meta_train, meta_valid = train_test_split(meta_data, test_size=validation_size, random_state=1234)
    logger.info('saving train split data to {}'.format(meta_train_split_filepath))
    meta_train.to_csv(meta_train_split_filepath, index=None)
    logger.info('saving valid split data to {}'.format(meta_valid_split_filepath))
    meta_valid.to_csv(meta_valid_split_filepath, index=None)


class Stacker(BaseTransformer):
    def __init__(self, columns_to_stack, target_columns, id_column):
        self.columns_to_stack = columns_to_stack
        self.target_columns = target_columns
        self.id_column = id_column

    def transform(self, meta, meta_valid=None, train_mode=True):
        meta_train_stacked = self._stack_columns(meta)
        X = meta_train_stacked['features'].values
        id = meta_train_stacked[self.id_column].values
        if train_mode:
            y = meta_train_stacked[self.target_columns].values
        else:
            y = None

        if meta_valid is not None:
            meta_valid_stacked = self._stack_columns(meta_valid)
            X_valid = meta_valid_stacked['features'].values
            y_valid = meta_valid_stacked[self.target_columns].values
            valid = X_valid, y_valid
        else:
            valid = None

        return {'X': X,
                'y': y,
                'id': id,
                'validation_data': valid,
                'train_mode': train_mode
                }

    def _stack_columns(self, df):
        return pd.melt(df,
                       id_vars=self.target_columns + self.id_column,
                       value_vars=self.columns_to_stack,
                       value_name='features')

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_stack = params['columns_to_stack']
        self.target_columns = params['target_columns']
        self.id_column = params['id_column']
        return self

    def save(self, filepath):
        params = {'columns_to_stack': self.columns_to_stack,
                  'target_columns': self.target_columns,
                  'id_column': self.id_column
                  }
        joblib.dump(params, filepath)


class ToNumpy(BaseTransformer):
    def __init__(self, columns_to_get, target_columns):
        self.columns_to_get = columns_to_get
        self.target_columns = target_columns

    def transform(self, meta, meta_valid=None, train_mode=True):
        X = meta[self.columns_to_get].values
        if train_mode:
            y = meta[self.target_columns].values
        else:
            y = None

        if meta_valid is not None:
            X_valid = meta_valid[self.columns_to_get].values
            y_valid = meta_valid[self.target_columns].values
            valid = X_valid, y_valid
        else:
            valid = None

        return {'X': X,
                'y': y,
                'validation_data': valid,
                'train_mode': train_mode}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['columns_to_get']
        self.target_columns = params['target_columns']
        return self

    def save(self, filepath):
        params = {'columns_to_get': self.columns_to_get,
                  'target_columns': self.target_columns
                  }
        joblib.dump(params, filepath)
