import os
from itertools import product

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from xgboost import XGBClassifier

from utils import read_predictions
from pipeline_config import Y_COLUMNS

SINGLE_DIR = '/mnt/ml-team/minerva/toxic/single_model_predictions_03092018'
RESULT_DIR = '/mnt/ml-team/minerva/debug'
GRID_SEARCH_RUNS = 100

param_grid = dict(objective=['rank:pairwise'],
                  eval_metric=['auc'],
                  scale_pos_weight=[100],
                  n_estimators=[500],
                  learning_rate=[0.1],
                  max_depth=[2, 3, 4, 5],
                  min_child_weight=[1, 3, 5, 7],
                  gamma=[0.01, 0.05, 0.1, 0.5],
                  subsample=[1.0, 0.8, 0.6],
                  colsample_bytree=[0.4, 0.6, 0.8, 1.0],
                  reg_lambda=[0.0, 0.01, 0.1, 0.5, 1.0],  # 1.0
                  reg_alpha=[0.0],
                  n_jobs=[12]
                  )


def get_fold_xy(train, test, label_columns, i):
    train_split = train[train['fold_id'] != i]
    valid_split = train[train['fold_id'] == i]
    test_split = test[test['fold_id'] == i]

    y_train = train_split[label_columns].values
    y_valid = valid_split[label_columns].values
    columns_to_drop_train = label_columns + ['id', 'fold_id']
    X_train = train_split.drop(columns_to_drop_train, axis=1).values
    X_valid = valid_split.drop(columns_to_drop_train, axis=1).values

    columns_to_drop_test = ['id', 'fold_id']
    X_test = test_split.drop(columns_to_drop_test, axis=1).values
    return (X_train, y_train), (X_valid, y_valid), X_test


def fit_cv(estimator, params, train, test, label_id, n_splits=10):
    estimators, scores, test_predictions = [], [], []
    for i in range(n_splits):
        (X_train, y_train), (X_valid, y_valid), X_test = get_fold_xy(train, test, Y_COLUMNS, i)

        y_train = y_train[:, label_id]
        y_valid = y_valid[:, label_id]

        estimator_ = estimator(**params)
        estimator_.fit(X_train, y_train,
                       early_stopping_rounds=10,
                       eval_metric=['error', 'auc'],
                       eval_set=[(X_train, y_train), (X_valid, y_valid)],
                       verbose=False)
        y_valid_pred = estimator_.predict_proba(X_valid, ntree_limit=estimator_.best_ntree_limit)[:, 1]
        y_test_pred = estimator_.predict_proba(X_test, ntree_limit=estimator_.best_ntree_limit)[:, 1]
        score = roc_auc_score(y_valid, y_valid_pred)
        estimators.append(estimator_)
        scores.append(score)
        test_predictions.append(y_test_pred)
    return scores, estimators, test_predictions


def make_grid(param_grid):
    keys, values = zip(*param_grid.items())
    param_dicts = [dict(zip(keys, v)) for v in product(*values)]
    return param_dicts


if __name__ == "__main__":

    train, test = read_predictions(SINGLE_DIR)

    estimator = XGBClassifier

    for label_id, label in enumerate(Y_COLUMNS):
        label_id = 0
        grid_sample = np.random.choice(make_grid(param_grid), GRID_SEARCH_RUNS, replace=False)

        grid_scores = []
        for params in tqdm(grid_sample):
            scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)
            print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
            grid_scores.append((params, np.mean(scores)))

        joblib.dump(grid_scores, os.path.join(RESULT_DIR, '{}_grid.pkl'.format(label)))
