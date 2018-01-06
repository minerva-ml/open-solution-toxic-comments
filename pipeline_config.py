import os
from attrdict import AttrDict

from utils import read_yaml

neptune_config = read_yaml('neptune_config.yaml')

X_COLUMNS = ['comment_text']
Y_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': neptune_config.parameters.experiment_dir},
    'fill_na': {'na_columns': X_COLUMNS},
    'xy_split': {'x_columns': X_COLUMNS,
                 'y_columns': Y_COLUMNS
                 },
    'char_tokenizer': {'char_level': True,
                       'maxlen': neptune_config.parameters.maxlen_char,
                       'num_words': neptune_config.parameters.max_features_char
                       },
    'word_tokenizer': {'char_level': False,
                       'maxlen': neptune_config.parameters.maxlen_words,
                       'num_words': neptune_config.parameters.max_features_word
                       },
    'tfidf_char_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'char',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, neptune_config.parameters.char_ngram_max),
                              'max_features': neptune_config.parameters.max_features_char
                              },
    'tfidf_word_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'word',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, 1),
                              'max_features': neptune_config.parameters.max_features_word
                              },
    'glove_embeddings': {'pretrained_filepath': neptune_config.parameters.embedding_filepath,
                         'max_features': neptune_config.parameters.max_features_word,
                         'embedding_size': neptune_config.parameters.word_embedding_size
                         },
    'char_cnn_network': {'architecture_config': {'model_params': {'maxlen': neptune_config.parameters.maxlen_char,
                                                                  'max_features': neptune_config.parameters.max_features_char,
                                                                  'embedding_size': neptune_config.parameters.char_embedding_size
                                                                  },
                                                 'optimizer_params': {'lr': neptune_config.parameters.lr,
                                                                      },
                                                 },
                         'training_config': {'epochs': neptune_config.parameters.epochs_nr,
                                             'batch_size': neptune_config.parameters.batch_size_train,
                                             },
                         'callbacks_config': {'model_checkpoint': {
                             'filepath': os.path.join(neptune_config.parameters.experiment_dir, 'checkpoints',
                                                      'char_cnn_network',
                                                      'char_cnn_network.h5'),
                             'save_best_only': True,
                             'save_weights_only': False},
                             'lr_scheduler': {'gamma': neptune_config.parameters.gamma},
                             'early_stopping': {'patience': neptune_config.parameters.patience},
                             'neptune_monitor': {},
                         },
                         },
    'word_lstm_network': {
        'architecture_config': {'model_params': {'max_features': neptune_config.parameters.max_features_word,
                                                 'maxlen': neptune_config.parameters.maxlen_words,
                                                 'embedding_size': neptune_config.parameters.word_embedding_size
                                                 },
                                'optimizer_params': {'lr': neptune_config.parameters.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_config.parameters.epochs_nr,
                            'batch_size': neptune_config.parameters.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_config.parameters.experiment_dir, 'checkpoints', 'word_lstm_network',
                                     'word_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_config.parameters.gamma},
            'early_stopping': {'patience': neptune_config.parameters.patience},
            'neptune_monitor': {},
        },
        },
    'word_glove_cnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_config.parameters.max_features_word,
                                                 'maxlen': neptune_config.parameters.maxlen_words,
                                                 'embedding_size': neptune_config.parameters.word_embedding_size
                                                 },
                                'optimizer_params': {'lr': neptune_config.parameters.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_config.parameters.epochs_nr,
                            'batch_size': neptune_config.parameters.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_config.parameters.experiment_dir, 'checkpoints',
                                     'word_glove_cnn_network',
                                     'word_glove_cnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_config.parameters.gamma},
            'early_stopping': {'patience': neptune_config.parameters.patience},
            'neptune_monitor': {},
        },
    },
    'word_glove_dpcnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_config.parameters.max_features_word,
                                                 'maxlen': neptune_config.parameters.maxlen_words,
                                                 'embedding_size': neptune_config.parameters.word_embedding_size,
                                                 'filter_nr': neptune_config.parameters.filter_nr,
                                                 'kernel_size': neptune_config.parameters.kernel_size,
                                                 'l2_reg': neptune_config.parameters.l2_reg,
                                                 'repeat_block': neptune_config.parameters.repeat_block,
                                                 'use_prelu': neptune_config.parameters.use_prelu
                                                 },
                                'optimizer_params': {'lr': neptune_config.parameters.lr,
                                                     'momentum': neptune_config.parameters.momentum,
                                                     'nesterov': True
                                                     },
                                },
        'training_config': {'epochs': neptune_config.parameters.epochs_nr,
                            'shuffle': True,
                            'batch_size': neptune_config.parameters.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_config.parameters.experiment_dir, 'checkpoints',
                                     'word_glove_dpcnn_network',
                                     'word_glove_dpcnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_config.parameters.gamma},
            'early_stopping': {'patience': neptune_config.parameters.patience},
            'neptune_monitor': {},
        },
    },
    'word_glove_lstm_network': {
        'architecture_config': {'model_params': {'max_features': neptune_config.parameters.max_features_word,
                                                 'maxlen': neptune_config.parameters.maxlen_words,
                                                 'embedding_size': neptune_config.parameters.word_embedding_size
                                                 },
                                'optimizer_params': {'lr': neptune_config.parameters.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_config.parameters.epochs_nr,
                            'batch_size': neptune_config.parameters.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_config.parameters.experiment_dir, 'checkpoints',
                                     'word_glove_lstm_network',
                                     'word_glove_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_config.parameters.gamma},
            'early_stopping': {'patience': neptune_config.parameters.patience},
            'neptune_monitor': {},
        },
    },
    'logistic_regression_multilabel': {'label_nr': 6,
                                       'C': neptune_config.parameters.log_reg_c,
                                       'solver': 'sag',
                                       'n_jobs': neptune_config.parameters.num_workers,
                                       },
    'prediction_average': {'weights': None
                           }
})
