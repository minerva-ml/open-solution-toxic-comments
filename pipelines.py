from steps.base import Step, Dummy, stack_inputs, sparse_hstack_inputs
from steps.preprocessing import XYSplit, FillNA, TfidfVectorizer
from steps.postprocessing import PredictionAverage
from steps.models.keras.loaders import Tokenizer
from steps.models.keras.models import GloveEmbeddingsMatrix
from steps.models.sklearn.models import LogisticRegressionMultilabel

from utils import fetch_x_train, fetch_x_valid, join_valid
from models import CharCNN, WordTrainableLSTM, GloveLSTM, GloveCNN, GloveDPCNN


def char_cnn_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    network_output = char_cnn_train(config, preprocessed_input)
    return network_output


def char_cnn_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    network_output = char_cnn_inference(config, preprocessed_input)
    return network_output


def word_lstm_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    network_output = word_lstm_train(config, preprocessed_input)
    return network_output


def word_lstm_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    network_output = word_lstm_inference(config, preprocessed_input)
    return network_output


def glove_lstm_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    network_output = glove_lstm_train(config, preprocessed_input)
    return network_output


def glove_lstm_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    network_output = glove_lstm_inference(config, preprocessed_input)
    return network_output


def glove_cnn_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    network_output = glove_cnn_train(config, preprocessed_input)
    return network_output


def glove_cnn_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    network_output = glove_cnn_inference(config, preprocessed_input)
    return network_output

def glove_dpcnn_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    network_output = glove_dpcnn_train(config, preprocessed_input)
    return network_output


def glove_dpcnn_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    network_output = glove_dpcnn_inference(config, preprocessed_input)
    return network_output


def tfidf_logreg_train_pipeline(config):
    preprocessed_input = train_preprocessing(config)
    logreg_output = tfidf_log_reg(config, preprocessed_input)
    return logreg_output


def tfidf_logreg_inference_pipeline(config):
    preprocessed_input = inference_preprocessing(config)
    logreg_output = tfidf_log_reg(config, preprocessed_input)
    return logreg_output


def ensemble_train_pipeline(config):
    fill_na_x = Step(name='fill_na_x',
                     transformer=FillNA(**config.fill_na),
                     input_data=['input_ensemble'],
                     adapter={'X': ([('input_ensemble', 'meta')])},
                     cache_dirpath=config.env.cache_dirpath)
    xy_split = Step(name='xy_split',
                    transformer=XYSplit(**config.xy_split),
                    input_data=['input_ensemble'],
                    input_steps=[fill_na_x],
                    adapter={'meta': ([('fill_na_x', 'X')]),
                             'train_mode': ([('input_ensemble', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[xy_split],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[xy_split],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[xy_split],
                                 adapter={'text': ([('xy_split', 'X')], fetch_x_train),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[xy_split],
                                 adapter={'text': ([('xy_split', 'X')], fetch_x_train),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

    log_reg_multi = Step(name='log_reg_multi',
                         transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                         input_steps=[xy_split, tfidf_char_vectorizer, tfidf_word_vectorizer],
                         adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                         ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                  'y': ([('xy_split', 'y')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         cache_output=True)

    char_cnn = Step(name='char_cnn',
                    transformer=CharCNN(**config.char_cnn_network),
                    input_steps=[char_tokenizer, xy_split],
                    adapter={'X': ([('char_tokenizer', 'X')]),
                             'y': ([('xy_split', 'y')]),
                             },
                    cache_dirpath=config.env.cache_dirpath,
                    cache_output=True)
    word_lstm = Step(name='word_lstm',
                     transformer=WordTrainableLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, xy_split],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath,
                     cache_output=True)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.word_glove_lstm_network),
                      input_steps=[word_tokenizer, xy_split, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('xy_split', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    glove_cnn = Step(name='glove_cnn',
                     transformer=GloveCNN(**config.word_glove_cnn_network),
                     input_steps=[word_tokenizer, xy_split, glove_embeddings],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                              },
                     cache_dirpath=config.env.cache_dirpath,
                     cache_output=True)

    prediction_average = Step(name='prediction_average',
                              transformer=PredictionAverage(**config.prediction_average),
                              input_steps=[char_cnn, word_lstm, glove_lstm, glove_cnn, log_reg_multi],
                              adapter={'prediction_proba_list': (
                                  [('char_cnn', 'prediction_probability'),
                                   ('log_reg_multi', 'prediction_probability'),
                                   ('word_lstm', 'prediction_probability'),
                                   ('glove_cnn', 'prediction_probability'),
                                   ('glove_lstm', 'prediction_probability')], stack_inputs)},
                              cache_dirpath=config.env.cache_dirpath)

    average_ensemble_output = Step(name='average_ensemble_output',
                                   transformer=Dummy(),
                                   input_steps=[prediction_average],
                                   adapter={'y_pred': ([('prediction_average', 'prediction_probability')]), },
                                   cache_dirpath=config.env.cache_dirpath)
    return average_ensemble_output


def ensemble_inference_pipeline(config):
    average_ensemble_output = ensemble_train_pipeline(config)

    cached_output_steps = ['char_cnn', 'log_reg_multi', 'word_lstm', 'glove_cnn', 'glove_lstm']
    for step_name in cached_output_steps:
        step = average_ensemble_output.get_step(step_name)
        step.cache_output = False

    return average_ensemble_output


def train_preprocessing(config):
    fill_na_x = Step(name='fill_na_x',
                     transformer=FillNA(**config.fill_na),
                     input_data=['input'],
                     adapter={'X': ([('input', 'meta')])},
                     cache_dirpath=config.env.cache_dirpath)
    fill_na_x_valid = Step(name='fill_na_x_valid',
                           transformer=FillNA(**config.fill_na),
                           input_data=['input'],
                           adapter={'X': ([('input', 'meta_valid')])},
                           cache_dirpath=config.env.cache_dirpath)
    xy_split = Step(name='xy_split',
                    transformer=XYSplit(**config.xy_split),
                    input_data=['input'],
                    input_steps=[fill_na_x, fill_na_x_valid],
                    adapter={'meta': ([('fill_na_x', 'X')]),
                             'meta_valid': ([('fill_na_x_valid', 'X')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)
    return xy_split


def inference_preprocessing(config):
    fill_na_x = Step(name='fill_na_x',
                     transformer=FillNA(**config.fill_na),
                     input_data=['input'],
                     adapter={'X': ([('input', 'meta')])},
                     cache_dirpath=config.env.cache_dirpath)
    xy_split = Step(name='xy_split',
                    transformer=XYSplit(**config.xy_split),
                    input_data=['input'],
                    input_steps=[fill_na_x],
                    adapter={'meta': ([('fill_na_x', 'X')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)
    return xy_split


def tfidf_log_reg(config, preprocessed_input):
    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('xy_split', 'X')], fetch_x_train),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('xy_split', 'X')], fetch_x_train),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    log_reg_multi = Step(name='log_reg_multi',
                         transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                         input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                         adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                         ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                  'y': ([('xy_split', 'y')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    log_reg_output = Step(name='log_reg_output',
                          transformer=Dummy(),
                          input_steps=[log_reg_multi],
                          adapter={'y_pred': ([('log_reg_multi', 'prediction_probability')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return log_reg_output


def char_cnn_train(config, preprocessed_input):
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'X_valid': ([('xy_split', 'validation_data')], fetch_x_valid),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    network = Step(name='char_cnn',
                   transformer=CharCNN(**config.char_cnn_network),
                   input_steps=[char_tokenizer, preprocessed_input],
                   adapter={'X': ([('char_tokenizer', 'X')]),
                            'y': ([('xy_split', 'y')]),
                            'validation_data': (
                                [('char_tokenizer', 'X_valid'), ('xy_split', 'validation_data')], join_valid),
                            },
                   cache_dirpath=config.env.cache_dirpath)
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_cnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def char_cnn_inference(config, preprocessed_input):
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    network = Step(name='char_cnn',
                   transformer=CharCNN(**config.char_cnn_network),
                   input_steps=[char_tokenizer, preprocessed_input],
                   adapter={'X': ([('char_tokenizer', 'X')]),
                            'y': ([('xy_split', 'y')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_cnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def word_lstm_train(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'X_valid': ([('xy_split', 'validation_data')], fetch_x_valid),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    word_lstm = Step(name='word_lstm',
                     transformer=WordTrainableLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'validation_data': (
                                  [('word_tokenizer', 'X_valid'), ('xy_split', 'validation_data')], join_valid),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def word_lstm_inference(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    word_lstm = Step(name='word_lstm',
                     transformer=WordTrainableLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def glove_prepro_train(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'X_valid': ([('xy_split', 'validation_data')], fetch_x_valid),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_prepro_inference(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('xy_split', 'X')], fetch_x_train),
                                   'train_mode': ([('xy_split', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_lstm_train(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_train(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.word_glove_lstm_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('xy_split', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               'validation_data': (
                                   [('word_tokenizer', 'X_valid'), ('xy_split', 'validation_data')], join_valid),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_lstm],
                        adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_lstm_inference(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_inference(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.word_glove_lstm_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('xy_split', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_lstm],
                        adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_cnn_train(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_train(config, preprocessed_input)
    glove_cnn = Step(name='glove_cnn',
                     transformer=GloveCNN(**config.word_glove_cnn_network),
                     input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                              'validation_data': (
                                  [('word_tokenizer', 'X_valid'), ('xy_split', 'validation_data')], join_valid),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_cnn],
                        adapter={'y_pred': ([('glove_cnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_cnn_inference(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_inference(config, preprocessed_input)
    glove_cnn = Step(name='glove_cnn',
                     transformer=GloveCNN(**config.word_glove_cnn_network),
                     input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_cnn],
                        adapter={'y_pred': ([('glove_cnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output

def glove_dpcnn_train(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_train(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                     transformer=GloveDPCNN(**config.word_glove_dpcnn_network),
                     input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                              'validation_data': (
                                  [('word_tokenizer', 'X_valid'), ('xy_split', 'validation_data')], join_valid),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_exp', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_dpcnn_inference(config, preprocessed_input):
    word_tokenizer, glove_embeddings = glove_prepro_inference(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                     transformer=GloveDPCNN(**config.word_glove_dpcnn_network),
                     input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_exp', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


PIPELINES = {'char_cnn': {'train': char_cnn_train_pipeline,
                          'inference': char_cnn_inference_pipeline},
             'word_trainable_lstm': {'train': word_lstm_train_pipeline,
                                     'inference': word_lstm_inference_pipeline},
             'glove_lstm': {'train': glove_lstm_train_pipeline,
                            'inference': glove_lstm_inference_pipeline},
             'glove_cnn': {'train': glove_cnn_train_pipeline,
                           'inference': glove_cnn_inference_pipeline},
             'glove_dpcnn': {'train': glove_dpcnn_train_pipeline,
                           'inference': glove_dpcnn_inference_pipeline},
             'tfidf_logreg': {'train': tfidf_logreg_train_pipeline,
                              'inference': tfidf_logreg_inference_pipeline},
             'weighted_average': {'train': ensemble_train_pipeline,
                                  'inference': ensemble_inference_pipeline},
             }
