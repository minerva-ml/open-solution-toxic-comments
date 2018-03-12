import glob
import logging
import os
from functools import reduce

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from sklearn.metrics import roc_auc_score, log_loss


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('toxic')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler()
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)


def get_logger():
    return logging.getLogger('toxic')


def read_data(data_dir, filename):
    meta_filepath = os.path.join(data_dir, filename)
    meta_data = pd.read_csv(meta_filepath)
    return meta_data


def read_predictions(prediction_dir, concat_mode='concat'):
    labels = pd.read_csv(os.path.join(prediction_dir, 'labels.csv'))

    filepaths_train, filepaths_test = [], []
    for filepath in sorted(glob.glob('{}/*'.format(prediction_dir))):
        if filepath.endswith('predictions_train_oof.csv'):
            filepaths_train.append(filepath)
        elif filepath.endswith('predictions_test_oof.csv'):
            filepaths_test.append(filepath)

    train_dfs = []
    for filepath in filepaths_train:
        train_dfs.append(pd.read_csv(filepath))
    train_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'fold_id']), train_dfs)
    train_dfs.columns = _clean_columns(train_dfs, keep_colnames = ['id','fold_id'])
    train_dfs = pd.merge(train_dfs, labels, on=['id'])

    test_dfs = []
    for filepath in filepaths_test:
        test_dfs.append(pd.read_csv(filepath))
    test_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'fold_id']), test_dfs)
    test_dfs.columns = _clean_columns(test_dfs, keep_colnames = ['id','fold_id'])

    return train_dfs, test_dfs


def _clean_columns(df, keep_colnames):
    new_colnames = []
    for i,colname in enumerate(df.columns):
        if colname not in keep_colnames:
            new_colnames.append(i)
        else:
            new_colnames.append(colname)
    return new_colnames


def create_predictions_df(meta, predictions, columns):
    submission = meta[['id']]
    predictions_ = pd.DataFrame(predictions, columns=columns)
    submission.reset_index(drop=True, inplace=True)
    predictions_.reset_index(drop=True, inplace=True)
    submission = pd.concat([submission, predictions_], axis=1)
    return submission


def save_submission(submission, experiments_dir, filename, logger):
    logger.info('submission head \n\n {}'.format(submission.head()))

    submission_filepath = os.path.join(experiments_dir, filename)
    submission.to_csv(submission_filepath, index=None)
    logger.info('submission saved to {}'.format(submission_filepath))


def create_submission(experiments_dir, filename, meta, predictions, columns, logger):
    submission_df = create_predictions_df(meta, predictions, columns)
    save_submission(submission_df, experiments_dir, filename, logger)


def multi_log_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(log_loss(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()
