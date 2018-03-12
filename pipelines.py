from functools import partial

from models import CharVDCNN, WordSCNN, WordDPCNN, WordCuDNNGRU, WordCuDNNLSTM, StackerRNN
from postprocessing import Blender
from steps.base import Step, Dummy, sparse_hstack_inputs, to_tuple_inputs
from steps.keras.loaders import Tokenizer
from steps.keras.models import GloveEmbeddingsMatrix, Word2VecEmbeddingsMatrix, FastTextEmbeddingsMatrix
from steps.preprocessing import XYSplit, TextCleaner, TfidfVectorizer, WordListFilter, Normalizer, TextCounter, \
    MinMaxScaler, MinMaxScalerMultilabel
from steps.sklearn.models import LogisticRegressionMultilabel, CatboostClassifierMultilabel, XGBoostClassifierMultilabel


def tfidf_logreg(config):
    preprocessed_input = _preprocessing(config, is_train=False)
    tfidf_char_vectorizer, tfidf_word_vectorizer = _tfidf(preprocessed_input, config)

    tfidf_logreg = Step(name='tfidf_logreg',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                        adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                        ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                 'y': ([('cleaning_output', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    output = Step(name='tfidf_logreg_output',
                  transformer=Dummy(),
                  input_steps=[tfidf_logreg],
                  adapter={'y_pred': ([('tfidf_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def bad_word_logreg(config):
    preprocessed_input = _preprocessing(config, is_train=False)
    tfidf_word_vectorizer = _bad_word_tfidf(preprocessed_input, config)

    bad_word_logreg = Step(name='bad_word_logreg',
                           transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                           input_steps=[preprocessed_input, tfidf_word_vectorizer],
                           adapter={'X': ([('bad_word_tfidf_word_vectorizer', 'features')]),
                                    'y': ([('cleaning_output', 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    output = Step(name='bad_word_logreg_output',
                  transformer=Dummy(),
                  input_steps=[bad_word_logreg],
                  adapter={'y_pred': ([('bad_word_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def count_features_logreg(config):
    normalizer = _count_features(config)
    xy_split = normalizer.get_step('xy_split')

    count_logreg = Step(name='count_logreg',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[xy_split, normalizer],
                        adapter={'X': ([('normalizer', 'X')]),
                                 'y': ([('xy_split', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    output = Step(name='count_logreg_output',
                  transformer=Dummy(),
                  input_steps=[count_logreg],
                  adapter={'y_pred': ([('count_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def bad_word_count_features_logreg(config):
    preprocessed_input = _preprocessing(config, is_train=False)
    normalizer = _count_features(config)
    xy_split = normalizer.get_step('xy_split')
    tfidf_word_vectorizer = _bad_word_tfidf(preprocessed_input, config)

    bad_word_count_logreg = Step(name='bad_word_count_logreg',
                                 transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                                 input_steps=[xy_split, normalizer, tfidf_word_vectorizer],
                                 adapter={'X': ([('normalizer', 'X'),
                                                 ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                          'y': ([('xy_split', 'y')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    output = Step(name='bad_word_count_features_logreg_output',
                  transformer=Dummy(),
                  input_steps=[bad_word_count_logreg],
                  adapter={'y_pred': ([('bad_word_count_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def hand_crafted_all_logreg(config):
    preprocessed_input = _preprocessing(config, is_train=False)
    tfidf_char_vectorizer, tfidf_word_vectorizer = _tfidf(preprocessed_input, config)
    normalizer = _count_features(config)
    xy_split = normalizer.get_step('xy_split')
    bad_word_vectorizer = _bad_word_tfidf(preprocessed_input, config)

    all_handcrafted_logreg = Step(name='all_handcrafted_logreg',
                                  transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                                  input_steps=[xy_split,
                                               normalizer,
                                               tfidf_char_vectorizer,
                                               tfidf_word_vectorizer,
                                               bad_word_vectorizer],
                                  adapter={'X': ([('normalizer', 'X'),
                                                  ('tfidf_char_vectorizer', 'features'),
                                                  ('tfidf_word_vectorizer', 'features'),
                                                  ('bad_word_tfidf_word_vectorizer', 'features')],
                                                 sparse_hstack_inputs),
                                           'y': ([('xy_split', 'y')]),
                                           },
                                  cache_dirpath=config.env.cache_dirpath)

    output = Step(name='hand_crafted_all_logreg_output',
                  transformer=Dummy(),
                  input_steps=[all_handcrafted_logreg],
                  adapter={'y_pred': ([('all_handcrafted_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def char_vdcnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    char_tokenizer = _char_tokenizer(preprocessed_input, config, is_train)

    if is_train:
        network = Step(name='char_vdcnn',
                       transformer=CharVDCNN(**config.char_vdcnn_network),
                       overwrite_transformer=True,
                       input_steps=[char_tokenizer, preprocessed_input],
                       adapter={'X': ([('char_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'validation_data': (
                                    [('char_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    else:
        network = Step(name='char_vdcnn',
                       transformer=CharVDCNN(**config.char_vdcnn_network),
                       input_steps=[char_tokenizer, preprocessed_input],
                       adapter={'X': ([('char_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    output = Step(name='char_vdcnn_output',
                  transformer=Dummy(),
                  input_steps=[network],
                  adapter={'y_pred': ([('char_vdcnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def glove_gru(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    glove_embeddings = _glove_embeddings(word_tokenizer, config)
    if is_train:
        glove_gru = Step(name='glove_gru',
                         transformer=WordCuDNNGRU(**config.gru_network),
                         overwrite_transformer=True,
                         input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                         adapter={'X': ([('word_tokenizer', 'X')]),
                                  'y': ([('cleaning_output', 'y')]),
                                  'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                  'validation_data': (
                                      [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                      to_tuple_inputs),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        glove_gru = Step(name='glove_gru',
                         transformer=WordCuDNNGRU(**config.gru_network),
                         input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                         adapter={'X': ([('word_tokenizer', 'X')]),
                                  'y': ([('cleaning_output', 'y')]),
                                  'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    output = Step(name='glove_gru_output',
                  transformer=Dummy(),
                  input_steps=[glove_gru],
                  adapter={'y_pred': ([('glove_gru', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def glove_lstm(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    glove_embeddings = _glove_embeddings(word_tokenizer, config)
    if is_train:
        glove_lstm = Step(name='glove_lstm',
                          transformer=WordCuDNNLSTM(**config.lstm_network),
                          overwrite_transformer=True,
                          input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                          adapter={'X': ([('word_tokenizer', 'X')]),
                                   'y': ([('cleaning_output', 'y')]),
                                   'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                   'validation_data': (
                                       [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                       to_tuple_inputs),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    else:
        glove_lstm = Step(name='glove_lstm',
                          transformer=WordCuDNNLSTM(**config.lstm_network),
                          input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                          adapter={'X': ([('word_tokenizer', 'X')]),
                                   'y': ([('cleaning_output', 'y')]),
                                   'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    output = Step(name='glove_lstm_output',
                  transformer=Dummy(),
                  input_steps=[glove_lstm],
                  adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def glove_scnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    glove_embeddings = _glove_embeddings(word_tokenizer, config)
    if is_train:
        glove_scnn = Step(name='glove_scnn',
                          transformer=WordSCNN(**config.scnn_network),
                          overwrite_transformer=True,
                          input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                          adapter={'X': ([('word_tokenizer', 'X')]),
                                   'y': ([('cleaning_output', 'y')]),
                                   'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                   'validation_data': (
                                       [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                       to_tuple_inputs),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    else:
        glove_scnn = Step(name='glove_scnn',
                          transformer=WordSCNN(**config.scnn_network),
                          input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                          adapter={'X': ([('word_tokenizer', 'X')]),
                                   'y': ([('cleaning_output', 'y')]),
                                   'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    output = Step(name='glove_scnn_output',
                  transformer=Dummy(),
                  input_steps=[glove_scnn],
                  adapter={'y_pred': ([('glove_scnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def glove_dpcnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    glove_embeddings = _glove_embeddings(word_tokenizer, config)
    if is_train:
        glove_dpcnn = Step(name='glove_dpcnn',
                           transformer=WordDPCNN(**config.dpcnn_network),
                           overwrite_transformer=True,
                           input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                           adapter={'X': ([('word_tokenizer', 'X')]),
                                    'y': ([('cleaning_output', 'y')]),
                                    'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                    'validation_data': (
                                        [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                        to_tuple_inputs),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    else:
        glove_dpcnn = Step(name='glove_dpcnn',
                           transformer=WordDPCNN(**config.dpcnn_network),
                           input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                           adapter={'X': ([('word_tokenizer', 'X')]),
                                    'y': ([('cleaning_output', 'y')]),
                                    'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    output = Step(name='glove_dpcnn_output',
                  transformer=Dummy(),
                  input_steps=[glove_dpcnn],
                  adapter={'y_pred': ([('glove_dpcnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def fasttext_lstm(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    fasttext_embeddings = _fasttext_embeddings(word_tokenizer, config)
    if is_train:
        fasttext_lstm = Step(name='fasttext_lstm',
                             transformer=WordCuDNNLSTM(**config.lstm_network),
                             overwrite_transformer=True,
                             input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                      'validation_data': (
                                          [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                          to_tuple_inputs),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    else:
        fasttext_lstm = Step(name='fasttext_lstm',
                             transformer=WordCuDNNLSTM(**config.lstm_network),
                             input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    output = Step(name='fasttext_lstm_output',
                  transformer=Dummy(),
                  input_steps=[fasttext_lstm],
                  adapter={'y_pred': ([('fasttext_lstm', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def fasttext_gru(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    fasttext_embeddings = _fasttext_embeddings(word_tokenizer, config)
    if is_train:
        fasttext_gru = Step(name='fasttext_gru',
                            transformer=WordCuDNNGRU(**config.gru_network),
                            overwrite_transformer=True,
                            input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                            adapter={'X': ([('word_tokenizer', 'X')]),
                                     'y': ([('cleaning_output', 'y')]),
                                     'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                     'validation_data': (
                                         [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                         to_tuple_inputs),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    else:
        fasttext_gru = Step(name='fasttext_gru',
                            transformer=WordCuDNNGRU(**config.gru_network),
                            input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                            adapter={'X': ([('word_tokenizer', 'X')]),
                                     'y': ([('cleaning_output', 'y')]),
                                     'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    output = Step(name='fasttext_gru_output',
                  transformer=Dummy(),
                  input_steps=[fasttext_gru],
                  adapter={'y_pred': ([('fasttext_gru', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def fasttext_dpcnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    fasttext_embeddings = _fasttext_embeddings(word_tokenizer, config)
    if is_train:
        fasttext_dpcnn = Step(name='fasttext_dpcnn',
                              transformer=WordDPCNN(**config.dpcnn_network),
                              overwrite_transformer=True,
                              input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                              adapter={'X': ([('word_tokenizer', 'X')]),
                                       'y': ([('cleaning_output', 'y')]),
                                       'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                       'validation_data': (
                                           [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                           to_tuple_inputs),
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    else:
        fasttext_dpcnn = Step(name='fasttext_dpcnn',
                              transformer=WordDPCNN(**config.dpcnn_network),
                              input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                              adapter={'X': ([('word_tokenizer', 'X')]),
                                       'y': ([('cleaning_output', 'y')]),
                                       'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    output = Step(name='fasttext_dpcnn_output',
                  transformer=Dummy(),
                  input_steps=[fasttext_dpcnn],
                  adapter={'y_pred': ([('fasttext_dpcnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def fasttext_scnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    fasttext_embeddings = _fasttext_embeddings(word_tokenizer, config)
    if is_train:
        fasttext_scnn = Step(name='fasttext_scnn',
                             transformer=WordSCNN(**config.scnn_network),
                             overwrite_transformer=True,
                             input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                      'validation_data': (
                                          [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                          to_tuple_inputs),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    else:
        fasttext_scnn = Step(name='fasttext_scnn',
                             transformer=WordSCNN(**config.scnn_network),
                             input_steps=[word_tokenizer, preprocessed_input, fasttext_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('fasttext_embeddings', 'embeddings_matrix')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    output = Step(name='fasttext_scnn_output',
                  transformer=Dummy(),
                  input_steps=[fasttext_scnn],
                  adapter={'y_pred': ([('fasttext_scnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def word2vec_gru(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    word2vec_embeddings = _word2vec_embeddings(word_tokenizer, config)
    if is_train:
        word2vec_gru = Step(name='word2vec_gru',
                            transformer=WordCuDNNGRU(**config.gru_network),
                            overwrite_transformer=True,
                            input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                            adapter={'X': ([('word_tokenizer', 'X')]),
                                     'y': ([('cleaning_output', 'y')]),
                                     'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                     'validation_data': (
                                         [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                         to_tuple_inputs),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    else:
        word2vec_gru = Step(name='word2vec_gru',
                            transformer=WordCuDNNGRU(**config.gru_network),
                            input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                            adapter={'X': ([('word_tokenizer', 'X')]),
                                     'y': ([('cleaning_output', 'y')]),
                                     'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    output = Step(name='word2vec_gru_output',
                  transformer=Dummy(),
                  input_steps=[word2vec_gru],
                  adapter={'y_pred': ([('word2vec_gru', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def word2vec_lstm(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    word2vec_embeddings = _word2vec_embeddings(word_tokenizer, config)
    if is_train:
        word2vec_lstm = Step(name='word2vec_lstm',
                             transformer=WordCuDNNLSTM(**config.lstm_network),
                             overwrite_transformer=True,
                             input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                      'validation_data': (
                                          [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                          to_tuple_inputs),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    else:
        word2vec_lstm = Step(name='word2vec_lstm',
                             transformer=WordCuDNNLSTM(**config.lstm_network),
                             input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    output = Step(name='word2vec_lstm_output',
                  transformer=Dummy(),
                  input_steps=[word2vec_lstm],
                  adapter={'y_pred': ([('word2vec_lstm', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def word2vec_dpcnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    word2vec_embeddings = _word2vec_embeddings(word_tokenizer, config)
    if is_train:
        word2vec_dpcnn = Step(name='word2vec_dpcnn',
                              transformer=WordDPCNN(**config.dpcnn_network),
                              overwrite_transformer=True,
                              input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                              adapter={'X': ([('word_tokenizer', 'X')]),
                                       'y': ([('cleaning_output', 'y')]),
                                       'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                       'validation_data': (
                                           [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                           to_tuple_inputs),
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    else:
        word2vec_dpcnn = Step(name='word2vec_dpcnn',
                              transformer=WordDPCNN(**config.dpcnn_network),
                              input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                              adapter={'X': ([('word_tokenizer', 'X')]),
                                       'y': ([('cleaning_output', 'y')]),
                                       'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    output = Step(name='word2vec_dpcnn_output',
                  transformer=Dummy(),
                  input_steps=[word2vec_dpcnn],
                  adapter={'y_pred': ([('word2vec_dpcnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def word2vec_scnn(config, is_train):
    preprocessed_input = _preprocessing(config, is_train)
    word_tokenizer = _word_tokenizer(preprocessed_input, config, is_train)
    word2vec_embeddings = _word2vec_embeddings(word_tokenizer, config)
    if is_train:
        word2vec_scnn = Step(name='word2vec_scnn',
                             transformer=WordSCNN(**config.scnn_network),
                             overwrite_transformer=True,
                             input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                      'validation_data': (
                                          [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')],
                                          to_tuple_inputs),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    else:
        word2vec_scnn = Step(name='word2vec_scnn',
                             transformer=WordSCNN(**config.scnn_network),
                             input_steps=[word_tokenizer, preprocessed_input, word2vec_embeddings],
                             adapter={'X': ([('word_tokenizer', 'X')]),
                                      'y': ([('cleaning_output', 'y')]),
                                      'embedding_matrix': ([('word2vec_embeddings', 'embeddings_matrix')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    output = Step(name='word2vec_scnn_output',
                  transformer=Dummy(),
                  input_steps=[word2vec_scnn],
                  adapter={'y_pred': ([('word2vec_scnn', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def blender_ensemble(config, is_train):
    minmax_scaler = Step(name='minmax_scaler',
                         transformer=MinMaxScalerMultilabel(),
                         input_data=['input'],
                         adapter={'X': ([('input', 'X')])},
                         cache_dirpath=config.env.cache_dirpath)

    blender_ensemble = Step(name='blender_ensemble',
                            transformer=Blender(**config.blender_ensemble),
                            input_data=['input'],
                            input_steps=[minmax_scaler],
                            adapter={'X': ([('minmax_scaler', 'X')]), 'y': ([('input', 'y')])},
                            cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[blender_ensemble],
                  adapter={'y_pred': ([('blender_ensemble', 'predictions')])},
                  cache_dirpath=config.env.cache_dirpath)

    if is_train:
        blender_ensemble.overwrite_transformer = True

    return output


def logreg_ensemble(config, is_train):
    minmax_scaler = Step(name='minmax_scaler',
                         transformer=MinMaxScaler(),
                         input_data=['input'],
                         adapter={'X': ([('input', 'X')])},
                         cache_dirpath=config.env.cache_dirpath)

    logreg_ensemble = Step(name='logreg_ensemble',
                           transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                           input_data=['input'],
                           input_steps=[minmax_scaler],
                           adapter={'X': ([('minmax_scaler', 'X')]), 'y': ([('input', 'y')])},
                           cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[logreg_ensemble],
                  adapter={'y_pred': ([('logreg_ensemble', 'prediction_probability')])},
                  cache_dirpath=config.env.cache_dirpath)

    if is_train:
        logreg_ensemble.overwrite_transformer = True

    return output


def catboost_ensemble(config, is_train):
    minmax_scaler = Step(name='minmax_scaler',
                         transformer=MinMaxScaler(),
                         input_data=['input'],
                         adapter={'X': ([('input', 'X')])},
                         cache_dirpath=config.env.cache_dirpath)

    catboost_ensemble = Step(name='catboost_ensemble',
                             transformer=CatboostClassifierMultilabel(**config.catboost_ensemble),
                             input_data=['input'],
                             input_steps=[minmax_scaler],
                             adapter={'X': ([('minmax_scaler', 'X')]), 'y': ([('input', 'y')])},
                             cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[catboost_ensemble],
                  adapter={'y_pred': ([('catboost_ensemble', 'prediction_probability')])},
                  cache_dirpath=config.env.cache_dirpath)

    if is_train:
        catboost_ensemble.overwrite_transformer = True

    return output


def xgboost_ensemble(config, is_train):
    minmax_scaler = Step(name='minmax_scaler',
                         transformer=MinMaxScaler(),
                         input_data=['input'],
                         adapter={'X': ([('input', 'X')])},
                         cache_dirpath=config.env.cache_dirpath)

    xgboost_ensemble = Step(name='xgboost_ensemble',
                            transformer=XGBoostClassifierMultilabel(**config.xgboost_ensemble),
                            input_data=['input'],
                            input_steps=[minmax_scaler],
                            adapter={'X': ([('minmax_scaler', 'X')]), 'y': ([('input', 'y')])},
                            cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[xgboost_ensemble],
                  adapter={'y_pred': ([('xgboost_ensemble', 'prediction_probability')])},
                  cache_dirpath=config.env.cache_dirpath)

    if is_train:
        xgboost_ensemble.overwrite_transformer = True

    return output


def rnn_ensemble(config, is_train):
    if is_train:
        minmax_scaler = Step(name='minmax_scaler',
                             transformer=MinMaxScalerMultilabel(),
                             input_data=['input'],
                             adapter={'X': ([('input', 'X')])},
                             cache_dirpath=config.env.cache_dirpath)

        minmax_scaler_valid_ = Step(name='minmax_scaler',
                                    transformer=MinMaxScalerMultilabel(),
                                    input_data=['input'],
                                    adapter={'X': ([('input', 'X_valid')])},
                                    cache_dirpath=config.env.cache_dirpath)

        minmax_scaler_valid = Step(name='minmax_scaler_valid',
                                   transformer=Dummy(),
                                   input_steps=[minmax_scaler_valid_],
                                   adapter={'X_valid': ([('minmax_scaler', 'X')]),
                                            },
                                   cache_dirpath=config.env.cache_dirpath)

        rnn_stacker_ensemble = Step(name='rnn_stacker_ensemble',
                                    transformer=StackerRNN(**config.rnn_stacker),
                                    input_data=['input'],
                                    input_steps=[minmax_scaler, minmax_scaler_valid],
                                    adapter={'X': ([('minmax_scaler', 'X')]),
                                             'y': ([('input', 'y')]),
                                             'validation_data': (
                                                 [('minmax_scaler_valid', 'X_valid'), ('input', 'y_valid')],
                                                 to_tuple_inputs),
                                             },
                                    cache_dirpath=config.env.cache_dirpath)
    else:
        minmax_scaler = Step(name='minmax_scaler',
                             transformer=MinMaxScalerMultilabel(),
                             input_data=['input'],
                             adapter={'X': ([('input', 'X')])},
                             cache_dirpath=config.env.cache_dirpath)

        rnn_stacker_ensemble = Step(name='rnn_stacker_ensemble',
                                    transformer=StackerRNN(**config.rnn_stacker),
                                    input_data=['input'],
                                    input_steps=[minmax_scaler],
                                    adapter={'X': ([('minmax_scaler', 'X')]),
                                             'y': ([('input', 'y')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath)
    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[rnn_stacker_ensemble],
                  adapter={'y_pred': ([('rnn_stacker_ensemble', 'prediction_probability')])},
                  cache_dirpath=config.env.cache_dirpath)

    if is_train:
        rnn_stacker_ensemble.overwrite_transformer = True

    return output


def _preprocessing(config, is_train=True):
    if is_train:
        xy_train = Step(name='xy_train',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

        text_cleaner_train = Step(name='text_cleaner_train',
                                  transformer=TextCleaner(**config.text_cleaner),
                                  input_steps=[xy_train],
                                  adapter={'X': ([('xy_train', 'X')])},
                                  cache_dirpath=config.env.cache_dirpath)

        xy_valid = Step(name='xy_valid',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

        text_cleaner_valid = Step(name='text_cleaner_valid',
                                  transformer=TextCleaner(**config.text_cleaner),
                                  input_steps=[xy_valid],
                                  adapter={'X': ([('xy_valid', 'X')])},
                                  cache_dirpath=config.env.cache_dirpath)

        cleaning_output = Step(name='cleaning_output',
                               transformer=Dummy(),
                               input_data=['input'],
                               input_steps=[xy_train, text_cleaner_train, xy_valid, text_cleaner_valid],
                               adapter={'X': ([('text_cleaner_train', 'X')]),
                                        'y': ([('xy_train', 'y')]),
                                        'train_mode': ([('input', 'train_mode')]),
                                        'X_valid': ([('text_cleaner_valid', 'X')]),
                                        'y_valid': ([('xy_valid', 'y')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)
    else:
        xy_train = Step(name='xy_train',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

        text_cleaner = Step(name='text_cleaner_train',
                            transformer=TextCleaner(**config.text_cleaner),
                            input_steps=[xy_train],
                            adapter={'X': ([('xy_train', 'X')])},
                            cache_dirpath=config.env.cache_dirpath)

        cleaning_output = Step(name='cleaning_output',
                               transformer=Dummy(),
                               input_data=['input'],
                               input_steps=[xy_train, text_cleaner],
                               adapter={'X': ([('text_cleaner_train', 'X')]),
                                        'y': ([('xy_train', 'y')]),
                                        'train_mode': ([('input', 'train_mode')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)
    return cleaning_output


def _char_tokenizer(preprocessed_input, config, is_train=True):
    if is_train:
        char_tokenizer = Step(name='char_tokenizer',
                              transformer=Tokenizer(**config.char_tokenizer),
                              input_steps=[preprocessed_input],
                              adapter={'X': ([('cleaning_output', 'X')]),
                                       'X_valid': ([('cleaning_output', 'X_valid')]),
                                       'train_mode': ([('cleaning_output', 'train_mode')])
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    else:
        char_tokenizer = Step(name='char_tokenizer',
                              transformer=Tokenizer(**config.char_tokenizer),
                              input_steps=[preprocessed_input],
                              adapter={'X': ([('cleaning_output', 'X')]),
                                       'train_mode': ([('cleaning_output', 'train_mode')])
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    return char_tokenizer


def _word_tokenizer(preprocessed_input, config, is_train=True):
    if is_train:
        word_tokenizer = Step(name='word_tokenizer',
                              transformer=Tokenizer(**config.word_tokenizer),
                              input_steps=[preprocessed_input],
                              adapter={'X': ([('cleaning_output', 'X')]),
                                       'train_mode': ([('cleaning_output', 'train_mode')]),
                                       'X_valid': ([('cleaning_output', 'X_valid')])
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    else:
        word_tokenizer = Step(name='word_tokenizer',
                              transformer=Tokenizer(**config.word_tokenizer),
                              input_steps=[preprocessed_input],
                              adapter={'X': ([('cleaning_output', 'X')]),
                                       'train_mode': ([('cleaning_output', 'train_mode')])
                                       },
                              cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer


def _glove_embeddings(word_tokenizer, config):
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([(word_tokenizer.name, 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return glove_embeddings


def _fasttext_embeddings(word_tokenizer, config):
    fasttext_embeddings = Step(name='fasttext_embeddings',
                               transformer=FastTextEmbeddingsMatrix(**config.embeddings),
                               input_steps=[word_tokenizer],
                               adapter={'tokenizer': ([(word_tokenizer.name, 'tokenizer')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)
    return fasttext_embeddings


def _word2vec_embeddings(word_tokenizer, config):
    word2vec_embeddings = Step(name='word2vec_embeddings',
                               transformer=Word2VecEmbeddingsMatrix(**config.embeddings),
                               input_steps=[word_tokenizer],
                               adapter={'tokenizer': ([(word_tokenizer.name, 'tokenizer')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)
    return word2vec_embeddings


def _tfidf(preprocessed_input, config):
    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    return tfidf_char_vectorizer, tfidf_word_vectorizer


def _bad_word_tfidf(preprocessed_input, config):
    bad_word_filter = Step(name='bad_word_filter',
                           transformer=WordListFilter(**config.bad_word_filter),
                           input_steps=[preprocessed_input],
                           adapter={'X': ([('cleaning_output', 'X')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)

    tfidf_word_vectorizer = Step(name='bad_word_tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[bad_word_filter],
                                 adapter={'text': ([('bad_word_filter', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    return tfidf_word_vectorizer


def _count_features(config):
    xy_split = Step(name='xy_split',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    text_counter = Step(name='text_counter',
                        transformer=TextCounter(),
                        input_steps=[xy_split],
                        adapter={'X': ([('xy_split', 'X')])},
                        cache_dirpath=config.env.cache_dirpath)

    normalizer = Step(name='normalizer',
                      transformer=Normalizer(),
                      input_steps=[text_counter],
                      adapter={'X': ([('text_counter', 'X')])},
                      cache_dirpath=config.env.cache_dirpath)

    return normalizer


PIPELINES = {'fasttext_gru': {'train': partial(fasttext_gru, is_train=True),
                              'inference': partial(fasttext_gru, is_train=False)},
             'fasttext_lstm': {'train': partial(fasttext_lstm, is_train=True),
                               'inference': partial(fasttext_lstm, is_train=False)},
             'fasttext_dpcnn': {'train': partial(fasttext_dpcnn, is_train=True),
                                'inference': partial(fasttext_dpcnn, is_train=False)},
             'fasttext_scnn': {'train': partial(fasttext_scnn, is_train=True),
                               'inference': partial(fasttext_scnn, is_train=False)},

             'word2vec_gru': {'train': partial(word2vec_gru, is_train=True),
                              'inference': partial(word2vec_gru, is_train=False)},
             'word2vec_lstm': {'train': partial(word2vec_lstm, is_train=True),
                               'inference': partial(word2vec_lstm, is_train=False)},
             'word2vec_dpcnn': {'train': partial(word2vec_dpcnn, is_train=True),
                                'inference': partial(word2vec_dpcnn, is_train=False)},
             'word2vec_scnn': {'train': partial(word2vec_scnn, is_train=True),
                               'inference': partial(word2vec_scnn, is_train=False)},

             'glove_gru': {'train': partial(glove_gru, is_train=True),
                           'inference': partial(glove_gru, is_train=False)},
             'glove_lstm': {'train': partial(glove_lstm, is_train=True),
                            'inference': partial(glove_lstm, is_train=False)},
             'glove_scnn': {'train': partial(glove_scnn, is_train=True),
                            'inference': partial(glove_scnn, is_train=False)},
             'glove_dpcnn': {'train': partial(glove_dpcnn, is_train=True),
                             'inference': partial(glove_dpcnn, is_train=False)},

             'char_vdcnn': {'train': partial(char_vdcnn, is_train=True),
                            'inference': partial(char_vdcnn, is_train=False)},

             'tfidf_logreg': {'train': tfidf_logreg,
                              'inference': tfidf_logreg},
             'bad_word_logreg': {'train': bad_word_logreg,
                                 'inference': bad_word_logreg},
             'count_logreg': {'train': count_features_logreg,
                              'inference': count_features_logreg},
             'bad_word_count_logreg': {'train': bad_word_count_features_logreg,
                                       'inference': bad_word_count_features_logreg},
             'hand_crafted_all_logreg': {'train': hand_crafted_all_logreg,
                                         'inference': hand_crafted_all_logreg},
             'blender_ensemble': {'train': partial(blender_ensemble, is_train=True),
                                  'inference': partial(blender_ensemble, is_train=False)},
             'logreg_ensemble': {'train': partial(logreg_ensemble, is_train=True),
                                 'inference': partial(logreg_ensemble, is_train=False)},
             'catboost_ensemble': {'train': partial(catboost_ensemble, is_train=True),
                                   'inference': partial(catboost_ensemble, is_train=False)},
             'xgboost_ensemble': {'train': partial(xgboost_ensemble, is_train=True),
                                  'inference': partial(xgboost_ensemble, is_train=False)},
             'rnn_ensemble': {'train': partial(rnn_ensemble, is_train=True),
                              'inference': partial(rnn_ensemble, is_train=False)},
             }
