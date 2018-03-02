import os

from langdetect import detect
from translation import bing
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import get_logger

logger = get_logger()


def split_train_data(data_dir, filename, target_columns, n_splits):
    meta_train_filepath = os.path.join(data_dir, filename)

    meta_train_split_filepath = meta_train_filepath.replace('train', 'train_split')
    meta_valid_split_filepath = meta_train_filepath.replace('train', 'valid_split')

    logger.info('reading data from {}'.format(meta_train_filepath))
    meta_data = pd.read_csv(meta_train_filepath).reset_index(drop=True)
    logger.info('splitting data')
    targets = meta_data[target_columns].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    skf.get_n_splits(targets)
    for train_idx, valid_idx in skf.split(targets, targets):
        meta_train = meta_data.iloc[train_idx]
        meta_valid = meta_data.iloc[valid_idx]
        logger.info('saving train split data to {}'.format(meta_train_split_filepath))
        meta_train.to_csv(meta_train_split_filepath, index=None)
        logger.info('saving valid split data to {}'.format(meta_valid_split_filepath))
        meta_valid.to_csv(meta_valid_split_filepath, index=None)
        break


def translate_data(data_dir, filename, filename_translated):
    meta_filepath = os.path.join(data_dir, filename)
    logger.info('reading data from {}'.format(meta_filepath))
    meta = pd.read_csv(meta_filepath)
    logger.info('detecting language')
    meta['lang'] = meta['comment_text'].apply(_safe_detect)
    logger.info('translating non english')
    meta['comment_text_english'] = meta.apply(_translate_non_english, axis=1)
    meta_translated_filepath = os.path.join(data_dir, filename_translated)
    logger.info('saving translated to{}'.format(meta_translated_filepath))
    meta.to_csv(meta, meta_translated_filepath, index=None)


def _safe_detect(text):
    try:
        lang = detect(text)
    except Exception:
        lang = 'en'
    return lang


def _translate_non_english(row):
    lang = row['lang']
    comment = row['comment_text']
    if lang != 'en':
        try:
            english_comment = bing(comment, dst='en')
        except Exception:
            english_comment = comment
    else:
        english_comment = comment
    return english_comment
