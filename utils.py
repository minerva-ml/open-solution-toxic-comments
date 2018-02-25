import logging
import os

import glob
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


def read_predictions(prediction_dir, mode='valid', valid_columns=None, stacking_mode='flat'):
    valid_labels = pd.read_csv(os.path.join(prediction_dir, 'valid_split.csv'))
    sample_submission = pd.read_csv(os.path.join(prediction_dir, 'sample_submission.csv'))
    predictions = []
    for filepath in sorted(glob.glob('{}/{}/*'.format(prediction_dir, mode))):
        prediction_single = pd.read_csv(filepath)
        prediction_single.drop('id', axis=1, inplace=True)
        predictions.append(prediction_single)

    if stacking_mode == 'flat':
        X = np.hstack(predictions)
    elif stacking_mode == 'rnn':
        X = np.stack(predictions, axis=2)
    else:
        raise NotImplementedError("""only stacking_mode options 'flat' and 'rnn' are supported""")

    if mode == 'valid':
        y = valid_labels[valid_columns].values
        return X, y
    elif mode == 'test':
        return X, sample_submission
    else:
        raise NotImplementedError


def create_submission(experiments_dir, filename, meta, predictions, columns, logger):
    submission = meta[['id']]
    predictions_ = pd.DataFrame(predictions, columns=columns)
    submission = pd.concat([submission, predictions_], axis=1)
    logger.info('submission head \n\n {}'.format(submission.head()))

    submission_filepath = os.path.join(experiments_dir, filename)
    submission.to_csv(submission_filepath, index=None)
    logger.info('submission saved to {}'.format(submission_filepath))


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
