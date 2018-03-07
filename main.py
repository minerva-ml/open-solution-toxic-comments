import os
import shutil

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from deepsense import neptune

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS, CV_LABELS
from pipelines import PIPELINES
from preprocessing import split_train_data, translate_data
from utils import init_logger, get_logger, read_params, read_data, read_predictions, multi_roc_auc_score, \
    create_submission, create_submission_df, save_submission

logger = get_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
def translate_to_english():
    logger.info('translating train')
    translate_data(data_dir=params.data_dir, filename='train.csv', filename_translated='train_translated.csv')
    logger.info('translating test')
    translate_data(data_dir=params.data_dir, filename='test.csv', filename_translated='test_translated.csv')


@action.command()
def train_valid_split():
    logger.info('preprocessing training data')
    split_train_data(data_dir=params.data_dir, filename='train_translated.csv', target_columns=CV_LABELS,
                     n_splits=params.n_cv_splits)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_pipeline(pipeline_name):
    _train_pipeline(pipeline_name)


def _train_pipeline(pipeline_name):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    train = read_data(data_dir=params.data_dir, filename='train_split_translated.csv')
    valid = read_data(data_dir=params.data_dir, filename='valid_split_translated.csv')

    data = {'input': {'meta': train,
                      'meta_valid': valid,
                      'train_mode': True,
                      },
            'input_ensemble': {'meta': valid,
                               'meta_valid': None,
                               'train_mode': True,
                               },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    output = pipeline.fit_transform(data)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def evaluate_pipeline(pipeline_name):
    _evaluate_pipeline(pipeline_name)


def _evaluate_pipeline(pipeline_name):
    valid = read_data(data_dir=params.data_dir, filename='valid_split_translated.csv')

    data = {'input': {'meta': valid,
                      'meta_valid': None,
                      'train_mode': False,
                      },
            'input_ensemble': {'meta': valid,
                               'meta_valid': None,
                               'train_mode': False,
                               },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_true = valid[Y_COLUMNS].values
    y_pred = output['y_pred']

    create_submission(params.experiment_dir, '{}_predictions_valid.csv'.format(pipeline_name), valid, y_pred, Y_COLUMNS,
                      logger)

    score = multi_roc_auc_score(y_true, y_pred)
    logger.info('Score on validation is {}'.format(score))
    ctx.channel_send('Final Validation Score ROC_AUC', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-m', '--model_level', help='first or second level', default='first', required=True)
@click.option('-s', '--stacking_mode', help='mode of stacking, flat or rnn', default='flat', required=False)
def predict_pipeline(pipeline_name, model_level, stacking_mode):
    _predict_pipeline(pipeline_name, model_level, stacking_mode)


def _predict_pipeline(pipeline_name, model_level, stacking_mode):
    if model_level == 'first':
        test = read_data(data_dir=params.data_dir, filename='test_translated.csv')
        data = {'input': {'meta': test,
                          'meta_valid': None,
                          'train_mode': False,
                          },
                }
    elif model_level == 'second':
        X, test = read_predictions(prediction_dir=params.single_model_predictions_dir,
                                   mode='test', stacking_mode=stacking_mode)
        data = {'input': {'X': X,
                          'y': None,
                          },
                }
    else:
        raise NotImplementedError("""only 'first' and 'second' """)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_pred = output['y_pred']

    create_submission(params.experiment_dir, '{}_predictions_test.csv'.format(pipeline_name),
                      test, y_pred, Y_COLUMNS, logger)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-m', '--model_level', help='first or second level', default='first', required=True)
@click.option('-s', '--stacking_mode', help='mode of stacking, flat or rnn', default='flat', required=False)
def train_evaluate_predict_pipeline(pipeline_name, model_level, stacking_mode):
    logger.info('training')
    _train_pipeline(pipeline_name)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)
    logger.info('predicting')
    _predict_pipeline(pipeline_name, model_level, stacking_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_evaluate_pipeline(pipeline_name):
    logger.info('training')
    _train_pipeline(pipeline_name)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-m', '--model_level', help='first or second level', default='first', required=True)
@click.option('-s', '--stacking_mode', help='mode of stacking, flat or rnn', default='flat', required=False)
def evaluate_predict_pipeline(pipeline_name, model_level, stacking_mode):
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)
    logger.info('predicting')
    _predict_pipeline(pipeline_name, model_level, stacking_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-m', '--model_level', help='first or second level', default='second', required=False)
@click.option('-s', '--stacking_mode', help='mode of stacking, flat or rnn', default='flat', required=False)
def train_evaluate_predict_cv_pipeline(pipeline_name, model_level, stacking_mode):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    if model_level == 'first':
        train = read_data(data_dir=params.data_dir, filename='train_translated.csv')#.sample(10000)
        train.reset_index(drop=True, inplace=True)
        cv_label = train[CV_LABELS].values

        test = read_data(data_dir=params.data_dir, filename='test_translated.csv')#.sample(100)
        test.reset_index(drop=True, inplace=True)
    elif model_level == 'second':
        X, y = read_predictions(prediction_dir=params.single_model_predictions_dir,
                                mode='valid', valid_columns=Y_COLUMNS, stacking_mode=stacking_mode)
        cv_label = y[:, 0]

        X_test, test = read_predictions(prediction_dir=params.single_model_predictions_dir,
                                        mode='test', stacking_mode=stacking_mode)

    else:
        raise NotImplementedError("""only 'first' and 'second' """)

    fold_scores, valid_predictions_out_of_fold, test_predictions_by_fold = [], [], []
    cv = StratifiedKFold(n_splits=params.n_cv_splits, shuffle=True, random_state=1234)
    cv.get_n_splits(cv_label)
    for i, (train_idx, valid_idx) in enumerate(cv.split(cv_label, cv_label)):
        logger.info('Fold {} started'.format(i))

        if model_level == 'first':
            train_split = train.iloc[train_idx]
            valid_split = train.iloc[valid_idx]
            y_true = valid_split[Y_COLUMNS].values

            data_train = {'input': {'meta': train_split,
                                    'meta_valid': valid_split,
                                    'train_mode': True,
                                    },
                          }
            data_valid = {'input': {'meta': valid_split,
                                    'meta_valid': None,
                                    'train_mode': False,
                                    }
                          }

            data_test = {'input': {'meta': test,
                                   'meta_valid': None,
                                   'train_mode': False,
                                   }
                         }
            logger.info('Training')
            pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
            output = pipeline.fit_transform(data_train)

            logger.info('Evaluation')
            pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
            output_valid = pipeline.transform(data_valid)
            y_valid_pred = output_valid['y_pred']
            valid_oof_submission = create_submission_df(valid_split, y_valid_pred, Y_COLUMNS)
            valid_predictions_out_of_fold.append(valid_oof_submission)
            logger.info('Saving fold {} oof predictions'.format(i))
            save_submission(valid_oof_submission, params.experiment_dir,
                            '{}_predictions_valid_fold{}.csv'.format(pipeline_name, i), logger)
            score = multi_roc_auc_score(y_true, y_valid_pred)
            logger.info('Score on fold {} is {}'.format(i, score))
            fold_scores.append(score)

            logger.info('Prediction')
            output_test = pipeline.transform(data_test)
            y_test_pred = output_test['y_pred']
            test_submission = create_submission_df(test, y_test_pred, Y_COLUMNS)
            test_predictions_by_fold.append(test_submission)
            logger.info('Saving fold {} test predictions'.format(i))
            save_submission(test_submission, params.experiment_dir,
                            '{}_predictions_test_fold{}.csv'.format(pipeline_name, i), logger)

        elif model_level == 'second':
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]

            y_true = y_valid

            data_train = {'input': {'X': X_train,
                                    'y': y_train,
                                    'X_valid': X_valid,
                                    'y_valid': y_valid
                                    },
                          }
            data_valid = {'input': {'X': X_valid,
                                    'y': y_valid,
                                    }
                          }

            data_test = {'input': {'X': X_test,
                                   'y': None,
                                   }
                         }

            logger.info('Training')
            pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
            output = pipeline.fit_transform(data_train)

            logger.info('Evaluation')
            pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
            output_valid = pipeline.transform(data_valid)
            y_valid_pred = output_valid['y_pred']
            valid_oof_submission = create_submission_df(valid_split, y_valid_pred, Y_COLUMNS)
            valid_predictions_out_of_fold.append(valid_oof_submission)
            logger.info('Saving fold {} oof predictions'.format(i))
            save_submission(valid_oof_submission, params.experiment_dir,
                            '{}_predictions_valid_fold{}.csv'.format(pipeline_name, i), logger)
            score = multi_roc_auc_score(y_true, y_valid_pred)
            logger.info('Score on fold {} is {}'.format(i, score))
            fold_scores.append(score)

            logger.info('Prediction')
            output_test = pipeline.transform(data_test)
            y_test_pred = output_test['y_pred']
            test_submission = create_submission_df(test, y_test_pred, Y_COLUMNS)
            test_predictions_by_fold.append(test_submission)
            logger.info('Saving fold {} test predictions'.format(i))
            save_submission(test_submission, params.experiment_dir,
                            '{}_predictions_test_fold{}.csv'.format(pipeline_name, i), logger)

        else:
            raise NotImplementedError("""only 'first' and 'second' """)

    mean_score = np.mean(fold_scores)
    logger.info('Score on validation is {}'.format(mean_score))
    ctx.channel_send('Final Validation Score ROC_AUC', 0, mean_score)

    if model_level == 'first':
        logger.info('Concatenating out of fold valid predictions')
        combined_oof_predictions = pd.concat(valid_predictions_out_of_fold, axis=0)
        save_submission(combined_oof_predictions, params.experiment_dir,
                        '{}_predictions_train_oof.csv'.format(pipeline_name), logger)
        logger.info('Averaging out of fold test predictions')
        test_predictions_by_fold = [prediction[Y_COLUMNS].values for prediction in test_predictions_by_fold]
        test_predictions_by_fold = np.stack(test_predictions_by_fold, axis=-1)
        mean_test_prediction = np.mean(test_predictions_by_fold, axis=-1)
        create_submission(params.experiment_dir, '{}_predictions_test_am.csv'.format(pipeline_name),
                          test, mean_test_prediction, Y_COLUMNS, logger)

    elif model_level == 'second':
        raise NotImplementedError('only first level model for now')


@action.command()
@click.argument('pipeline_names', nargs=-1)
def prepare_single_model_predictions_dir(pipeline_names):
    os.makedirs(params.single_model_predictions_dir, exist_ok=True)

    valid_split_source = os.path.join(params.data_dir, 'valid_split_translated.csv')
    valid_split_destination = os.path.join(params.single_model_predictions_dir, 'valid_split_translated.csv')
    logger.info('copying valid_split from {} to {}'.format(valid_split_source, valid_split_destination))
    shutil.copy(valid_split_source, valid_split_destination)

    sample_submit_source = os.path.join(params.data_dir, 'sample_submission.csv')
    sample_submit_destination = os.path.join(params.single_model_predictions_dir, 'sample_submission.csv')
    logger.info('copying valid_split from {} to {}'.format(sample_submit_source, sample_submit_destination))
    shutil.copy(sample_submit_source, sample_submit_destination)

    for pipeline_name in pipeline_names:
        pipeline_dir = os.path.join(params.experiment_dir, pipeline_name)
        for fold in ['valid', 'test']:
            fold_dirpath = os.path.join(params.single_model_predictions_dir, fold)
            os.makedirs(fold_dirpath, exist_ok=True)
            fold_filename = '{}_predictions_{}.csv'.format(pipeline_name, fold)
            source_filepath = os.path.join(pipeline_dir, fold_filename)
            destination_filepath = os.path.join(fold_dirpath, fold_filename)
            logger.info('copying {} from {} to {}'.format(fold, source_filepath, destination_filepath))
            shutil.copy(source_filepath, destination_filepath)


if __name__ == "__main__":
    init_logger()
    action()
