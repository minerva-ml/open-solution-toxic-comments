from models import CharVDCNN, WordLSTM, GloveLSTM, GloveSCNN, GloveDPCNN
from steps.base import Step, Dummy, hstack_inputs, sparse_hstack_inputs, to_tuple_inputs
from steps.keras.loaders import Tokenizer
from steps.keras.models import GloveEmbeddingsMatrix
from steps.preprocessing import XYSplit, TextCleaner, TfidfVectorizer, WordListFilter, Normalizer, TextCounter
from steps.sklearn.models import LogisticRegressionMultilabel, LinearSVCMultilabel, CatboostClassifierMultilabel


def train_preprocessing(config):
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
    return cleaning_output


def inference_preprocessing(config):
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


def char_vdcnn_train(config):
    preprocessed_input = train_preprocessing(config)
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
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
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_vdcnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def char_vdcnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    network = Step(name='char_vdcnn',
                   transformer=CharVDCNN(**config.char_vdcnn_network),
                   input_steps=[char_tokenizer, preprocessed_input],
                   adapter={'X': ([('char_tokenizer', 'X')]),
                            'y': ([('cleaning_output', 'y')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_vdcnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def word_lstm_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     overwrite_transformer=True,
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('cleaning_output', 'y')]),
                              'validation_data': (
                                  [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def word_lstm_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('cleaning_output', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def glove_preprocessing_train(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_preprocessing_inference(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_lstm_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      overwrite_transformer=True,
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               'validation_data': (
                                   [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_lstm],
                        adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_lstm_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
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


def glove_scnn_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      overwrite_transformer=True,
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               'validation_data': (
                                   [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_scnn],
                        adapter={'y_pred': ([('glove_scnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_scnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_scnn],
                        adapter={'y_pred': ([('glove_scnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_dpcnn_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       overwrite_transformer=True,
                       input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                'validation_data': (
                                    [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_dpcnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_dpcnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_dpcnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def tfidf(preprocessed_input, config):
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


def tfidf_logreg(config):
    preprocessed_input = inference_preprocessing(config)
    tfidf_char_vectorizer, tfidf_word_vectorizer = tfidf(preprocessed_input, config)

    logreg_tfidf = Step(name='logreg_tfidf',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                        adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                        ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                 'y': ([('cleaning_output', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    logreg_output = Step(name='logreg_output',
                         transformer=Dummy(),
                         input_steps=[logreg_tfidf],
                         adapter={'y_pred': ([('logreg_tfidf', 'prediction_probability')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return logreg_output


def tfidf_svm(config):
    preprocessed_input = inference_preprocessing(config)
    tfidf_char_vectorizer, tfidf_word_vectorizer = tfidf(preprocessed_input, config)

    svm_multi = Step(name='svm_multi',
                     transformer=LinearSVCMultilabel(**config.svc_multilabel),
                     input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                     adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                     ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                              'y': ([('cleaning_output', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    svm_output = Step(name='svm_output',
                      transformer=Dummy(),
                      input_steps=[svm_multi],
                      adapter={'y_pred': ([('logreg_multi', 'prediction_probability')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return svm_output


def bad_word_tfidf(preprocessed_input, config):
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


def bad_word_logreg(config):
    preprocessed_input = inference_preprocessing(config)
    tfidf_word_vectorizer = bad_word_tfidf(preprocessed_input, config)

    logreg_bad_word = Step(name='logreg_bad_word',
                           transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                           input_steps=[preprocessed_input, tfidf_word_vectorizer],
                           adapter={'X': ([('bad_word_tfidf_word_vectorizer', 'features')]),
                                    'y': ([('cleaning_output', 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    logreg_output = Step(name='logreg_output',
                         transformer=Dummy(),
                         input_steps=[logreg_bad_word],
                         adapter={'y_pred': ([('logreg_bad_word', 'prediction_probability')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return logreg_output


def bad_word_tfidf_svm(config):
    preprocessed_input = inference_preprocessing(config)
    tfidf_word_vectorizer = bad_word_tfidf(preprocessed_input, config)

    svm_multi = Step(name='svm_multi',
                     transformer=LinearSVCMultilabel(**config.svc_multilabel),
                     input_steps=[preprocessed_input, tfidf_word_vectorizer],
                     adapter={'X': ([('bad_word_tfidf_word_vectorizer', 'features')]),
                              'y': ([('cleaning_output', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    svm_output = Step(name='svm_output',
                      transformer=Dummy(),
                      input_steps=[svm_multi],
                      adapter={'y_pred': ([('svm_multi', 'prediction_probability')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return svm_output


def count_features(config):
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


def count_features_logreg(config):
    normalizer = count_features(config)
    xy_split = normalizer.get_step('xy_split')

    logreg_count = Step(name='logreg_count',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[xy_split, normalizer],
                        adapter={'X': ([('normalizer', 'X')]),
                                 'y': ([('xy_split', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    logreg_output = Step(name='logreg_output',
                         transformer=Dummy(),
                         input_steps=[logreg_count],
                         adapter={'y_pred': ([('logreg_count', 'prediction_probability')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return logreg_output


def count_features_svm(config):
    normalizer = count_features(config)
    xy_split = normalizer.get_step('xy_split')

    svm_multi = Step(name='svm_multi',
                     transformer=LinearSVCMultilabel(**config.svc_multilabel),
                     input_steps=[xy_split, normalizer],
                     adapter={'X': ([('normalizer', 'X')]),
                              'y': ([('xy_split', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)

    svm_output = Step(name='svm_output',
                      transformer=Dummy(),
                      input_steps=[svm_multi],
                      adapter={'y_pred': ([('svm_multi', 'prediction_probability')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return svm_output


def bad_word_count_features_logreg(config):
    preprocessed_input = inference_preprocessing(config)
    normalizer = count_features(config)
    xy_split = normalizer.get_step('xy_split')
    tfidf_word_vectorizer = bad_word_tfidf(preprocessed_input, config)

    logreg_bad_word_count = Step(name='logreg_bad_word_count',
                                 transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                                 input_steps=[xy_split, normalizer, tfidf_word_vectorizer],
                                 adapter={'X': ([('normalizer', 'X'),
                                                 ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                          'y': ([('xy_split', 'y')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    logreg_output = Step(name='logreg_output',
                         transformer=Dummy(),
                         input_steps=[logreg_bad_word_count],
                         adapter={'y_pred': ([('logreg_bad_word_count', 'prediction_probability')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return logreg_output


def bad_word_count_features_svm(config):
    preprocessed_input = inference_preprocessing(config)
    normalizer = count_features(config)
    xy_split = normalizer.get_step('xy_split')
    tfidf_word_vectorizer = bad_word_tfidf(preprocessed_input, config)

    svm_multi = Step(name='svm_multi',
                     transformer=LinearSVCMultilabel(**config.svc_multilabel),
                     input_steps=[xy_split, normalizer, tfidf_word_vectorizer],
                     adapter={'X': ([('normalizer', 'X'),
                                     ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                              'y': ([('xy_split', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)

    svm_output = Step(name='svm_output',
                      transformer=Dummy(),
                      input_steps=[svm_multi],
                      adapter={'y_pred': ([('svm_multi', 'prediction_probability')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return svm_output


def hand_crafted_all(config):
    preprocessed_input = inference_preprocessing(config)
    tfidf_char_vectorizer, tfidf_word_vectorizer = tfidf(preprocessed_input, config)
    normalizer = count_features(config)
    xy_split = normalizer.get_step('xy_split')
    bad_word_vectorizer = bad_word_tfidf(preprocessed_input, config)

    return xy_split, normalizer, tfidf_char_vectorizer, tfidf_word_vectorizer, bad_word_vectorizer


def hand_crafted_all_logreg(config):
    xy_split, normalizer, char_vector, word_vector, bad_word_vector = hand_crafted_all(config)

    logreg_multi = Step(name='logreg_multi',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[xy_split, normalizer, char_vector, word_vector, bad_word_vector],
                        adapter={'X': ([('normalizer', 'X'),
                                        ('tfidf_char_vectorizer', 'features'),
                                        ('tfidf_word_vectorizer', 'features'),
                                        ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                 'y': ([('xy_split', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    logreg_output = Step(name='logreg_output',
                         transformer=Dummy(),
                         input_steps=[logreg_multi],
                         adapter={'y_pred': ([('logreg_multi', 'prediction_probability')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return logreg_output


def hand_crafted_all_svm(config):
    xy_split, normalizer, char_vector, word_vector, bad_word_vector = hand_crafted_all(config)

    svm_multi = Step(name='svm_multi',
                     transformer=LinearSVCMultilabel(**config.svc_multilabel),
                     input_steps=[xy_split, normalizer, char_vector, word_vector, bad_word_vector],
                     adapter={'X': ([('normalizer', 'X'),
                                     ('tfidf_char_vectorizer', 'features'),
                                     ('tfidf_word_vectorizer', 'features'),
                                     ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                              'y': ([('xy_split', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)

    svm_output = Step(name='svm_output',
                      transformer=Dummy(),
                      input_steps=[svm_multi],
                      adapter={'y_pred': ([('logreg_multi', 'prediction_probability')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return svm_output


def ensemble_extraction(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input_ensemble'],
                    adapter={'meta': ([('input_ensemble', 'meta')]),
                             'train_mode': ([('input_ensemble', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)
    text_cleaner_train = Step(name='text_cleaner_train',
                              transformer=TextCleaner(**config.text_cleaner),
                              input_steps=[xy_train],
                              adapter={'X': ([('xy_train', 'X')])},
                              cache_dirpath=config.env.cache_dirpath)

    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[text_cleaner_train],
                          input_data=['input_ensemble'],
                          adapter={'X': ([('text_cleaner_train', 'X')]),
                                   'train_mode': ([('input_ensemble', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[text_cleaner_train],
                          input_data=['input_ensemble'],
                          adapter={'X': ([('text_cleaner_train', 'X')]),
                                   'train_mode': ([('input_ensemble', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[text_cleaner_train],
                                 adapter={'text': ([('text_cleaner_train', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[text_cleaner_train],
                                 adapter={'text': ([('text_cleaner_train', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    bad_word_filter = Step(name='bad_word_filter',
                           transformer=WordListFilter(**config.bad_word_filter),
                           input_steps=[text_cleaner_train],
                           adapter={'X': ([('text_cleaner_train', 'X')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)

    bad_word_tfidf_word_vectorizer = Step(name='bad_word_tfidf_word_vectorizer',
                                          transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                          input_steps=[bad_word_filter],
                                          adapter={'text': ([('bad_word_filter', 'X')]),
                                                   },
                                          cache_dirpath=config.env.cache_dirpath)

    text_counter = Step(name='text_counter',
                        transformer=TextCounter(),
                        input_steps=[xy_train],
                        adapter={'X': ([('xy_train', 'X')])},
                        cache_dirpath=config.env.cache_dirpath)

    normalizer = Step(name='normalizer',
                      transformer=Normalizer(),
                      input_steps=[text_counter],
                      adapter={'X': ([('text_counter', 'X')])},
                      cache_dirpath=config.env.cache_dirpath)

    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

    logreg_count = Step(name='logreg_count',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[xy_train, normalizer],
                        adapter={'X': ([('normalizer', 'X')]),
                                 'y': ([('xy_train', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)
    logreg_bad_word = Step(name='logreg_bad_word',
                           transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                           input_steps=[xy_train, bad_word_tfidf_word_vectorizer],
                           adapter={'X': ([('bad_word_tfidf_word_vectorizer', 'features')]),
                                    'y': ([('xy_train', 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           cache_output=True)
    logreg_bad_word_count = Step(name='logreg_bad_word_count',
                                 transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                                 input_steps=[xy_train, normalizer, bad_word_tfidf_word_vectorizer],
                                 adapter={'X': ([('normalizer', 'X'),
                                                 ('bad_word_tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                          'y': ([('xy_train', 'y')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 cache_output=True)
    logreg_tfidf = Step(name='logreg_tfidf',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[xy_train, tfidf_char_vectorizer, tfidf_word_vectorizer],
                        adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                        ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                 'y': ([('xy_train', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)
    char_vdcnn = Step(name='char_vdcnn',
                      transformer=CharVDCNN(**config.char_vdcnn_network),
                      input_steps=[char_tokenizer, xy_train],
                      adapter={'X': ([('char_tokenizer', 'X')]),
                               'y': ([('xy_train', 'y')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, xy_train],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('xy_train', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath,
                     cache_output=True)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      input_steps=[word_tokenizer, xy_train, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('xy_train', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      input_steps=[word_tokenizer, xy_train, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('xy_train', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       input_steps=[word_tokenizer, xy_train, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('xy_train', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       cache_output=True)

    return [logreg_count, logreg_bad_word, logreg_bad_word_count,
            logreg_tfidf, char_vdcnn, word_lstm, glove_lstm,
            glove_scnn, glove_dpcnn]


def catboost_ensemble_train(config):
    model_outputs = ensemble_extraction(config)
    output_mappings = [(output_step.name, 'prediction_probability') for output_step in model_outputs]

    label = model_outputs[0].get_step('xy_train')

    input_steps = model_outputs + [label]

    catboost_ensemble = Step(name='catboost_ensemble',
                             transformer=CatboostClassifierMultilabel(**config.catboost_ensemble),
                             overwrite_transformer=True,
                             input_steps=input_steps,
                             adapter={'X': (output_mappings, hstack_inputs),
                                      'y': ([('xy_train', 'y')])},
                             cache_dirpath=config.env.cache_dirpath)

    catboost_ensemble_output = Step(name='catboost_ensemble_output',
                                    transformer=Dummy(),
                                    input_steps=[catboost_ensemble],
                                    adapter={'y_pred': ([('catboost_ensemble', 'prediction_probability')])},
                                    cache_dirpath=config.env.cache_dirpath)
    return catboost_ensemble_output


def catboost_ensemble_inference(config):
    catboost_ensemble = catboost_ensemble_train(config)
    ensemble_step = catboost_ensemble.get_step('catboost_ensemble')
    ensemble_step.overwrite_transformer = False
    for step in ensemble_step.input_steps:
        step.cache_output = False
    return catboost_ensemble


PIPELINES = {'char_vdcnn': {'train': char_vdcnn_train,
                            'inference': char_vdcnn_inference},
             'word_lstm': {'train': word_lstm_train,
                           'inference': word_lstm_inference},
             'glove_lstm': {'train': glove_lstm_train,
                            'inference': glove_lstm_inference},
             'glove_scnn': {'train': glove_scnn_train,
                            'inference': glove_scnn_inference},
             'glove_dpcnn': {'train': glove_dpcnn_train,
                             'inference': glove_dpcnn_inference},
             'tfidf_logreg': {'train': tfidf_logreg,
                              'inference': tfidf_logreg},
             'tfidf_svm': {'train': tfidf_svm,
                           'inference': tfidf_svm},
             'bad_word_logreg': {'train': bad_word_logreg,
                                 'inference': bad_word_logreg},
             'bad_word_tfidf_svm': {'train': bad_word_tfidf_svm,
                                    'inference': bad_word_tfidf_svm},
             'count_logreg': {'train': count_features_logreg,
                              'inference': count_features_logreg},
             'count_svm': {'train': count_features_svm,
                           'inference': count_features_svm},
             'bad_word_count_logreg': {'train': bad_word_count_features_logreg,
                                       'inference': bad_word_count_features_logreg},
             'bad_word_count_svm': {'train': bad_word_count_features_svm,
                                    'inference': bad_word_count_features_svm},
             'hand_crafted_all_svm': {'train': hand_crafted_all_svm,
                                      'inference': hand_crafted_all_svm},
             'catboost_ensemble': {'train': catboost_ensemble_train,
                                   'inference': catboost_ensemble_inference},
             }
