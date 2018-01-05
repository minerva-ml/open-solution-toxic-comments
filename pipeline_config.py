import os
from attrdict import AttrDict

from utils import read_yaml

neptune_config = read_yaml('neptune_config.yaml')

X_COLUMNS = ['comment_text']
Y_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

GLOBAL_CONFIG = {'exp_root': neptune_config.parameters.experiment_dir,
                 'num_workers': 6,
                 'max_features': 100000,
                 'maxlen_char': 512,
                 'maxlen_words': 64,
                 'batch_size_train': 128,
                 'batch_size_inference': 128,
                 }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': GLOBAL_CONFIG['exp_root']},
    'fill_na': {'na_columns': X_COLUMNS},
    'xy_split': {'x_columns': X_COLUMNS,
                 'y_columns': Y_COLUMNS
                 },
    'char_tokenizer': {'char_level': True,
                       'maxlen': GLOBAL_CONFIG['maxlen_char']
                       },
    'word_tokenizer': {'char_level': False,
                       'maxlen': GLOBAL_CONFIG['maxlen_words']
                       },
    'tfidf_char_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'char',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, 4),
                              'max_features': GLOBAL_CONFIG['max_features']
                              },
    'tfidf_word_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'word',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, 1),
                              'max_features': GLOBAL_CONFIG['max_features']
                              },
    'glove_embeddings': {'pretrained_filepath': neptune_config.parameters.embedding_filepath,
                         'max_features': GLOBAL_CONFIG['max_features'],
                         'embedding_size': 300
                         },
    'char_cnn_network': {'architecture_config': {'model_params': {'maxlen': GLOBAL_CONFIG['maxlen_char'],
                                                                  'max_features': 1989,
                                                                  'embedding_size': 256
                                                                  },
                                                 'optimizer_params': {'lr': 0.00025,
                                                                      },
                                                 },
                         'training_config': {'epochs': 1000,
                                             'batch_size': GLOBAL_CONFIG['batch_size_train'],
                                             },
                         'callbacks_config': {'model_checkpoint': {
                             'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'char_cnn_network',
                                                      'char_cnn_network_best.h5'),
                             'save_best_only': True,
                             'save_weights_only': False},
                             'lr_scheduler': {'gamma': 0.99},
                             'early_stopping': {'patience': 5},
                             'neptune_monitor': {},
                         },
                         },
    'word_lstm_network': {'architecture_config': {'model_params': {'max_features': GLOBAL_CONFIG['max_features'],
                                                                   'maxlen': GLOBAL_CONFIG['maxlen_words'],
                                                                   'embedding_size': 256
                                                                   },
                                                  'optimizer_params': {'lr': 0.00025,
                                                                       },
                                                  },
                          'training_config': {'epochs': 1000,
                                              'batch_size': GLOBAL_CONFIG['batch_size_train'],
                                              },
                          'callbacks_config': {'model_checkpoint': {
                              'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'word_lstm_network',
                                                       'word_lstm_network_best.h5'),
                              'save_best_only': True,
                              'save_weights_only': False},
                              'lr_scheduler': {'gamma': 0.99},
                              'early_stopping': {'patience': 5},
                              'neptune_monitor': {},
                          },
                          },
    'word_glove_cnn_network': {
        'architecture_config': {'model_params': {'max_features': GLOBAL_CONFIG['max_features'],
                                                 'maxlen': GLOBAL_CONFIG['maxlen_words'],
                                                 'embedding_size': 300
                                                 },
                                'optimizer_params': {'lr': 0.00025,
                                                     },
                                },
        'training_config': {'epochs': 1000,
                            'batch_size': GLOBAL_CONFIG['batch_size_train'],
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints',
                                     'word_glove_lstm_network',
                                     'word_glove_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': 0.99},
            'early_stopping': {'patience': 10},
            'neptune_monitor': {},
        },
    },
    'word_glove_dpcnn_network': {
        'architecture_config': {'model_params': {'max_features': GLOBAL_CONFIG['max_features'],
                                                 'maxlen': GLOBAL_CONFIG['maxlen_words'],
                                                 'embedding_size': 300
                                                 },
                                'optimizer_params': {'lr': 0.00025,
                                                     },
                                },
        'training_config': {'epochs': 1000,
                            'batch_size': GLOBAL_CONFIG['batch_size_train'],
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints',
                                     'word_glove_lstm_network',
                                     'word_glove_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': 0.99},
            'early_stopping': {'patience': 10},
            'neptune_monitor': {},
        },
    },
    'word_glove_lstm_network': {
        'architecture_config': {'model_params': {'max_features': GLOBAL_CONFIG['max_features'],
                                                 'maxlen': GLOBAL_CONFIG['maxlen_words'],
                                                 'embedding_size': 300
                                                 },
                                'optimizer_params': {'lr': 0.00025,
                                                     },
                                },
        'training_config': {'epochs': 1000,
                            'batch_size': GLOBAL_CONFIG['batch_size_train'],
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints',
                                     'word_glove_lstm_network',
                                     'word_glove_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': 0.99},
            'early_stopping': {'patience': 10},
            'neptune_monitor': {},
        },
    },
    'logistic_regression_multilabel': {'label_nr': 6,
                                       'C': 4.0,
                                       'solver': 'sag',
                                       'n_jobs': -2,
                                       },
    'prediction_average': {'weights': None
                           }
})
