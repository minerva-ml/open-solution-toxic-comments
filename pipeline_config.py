import os

from attrdict import AttrDict
from deepsense import neptune

# from utils import read_yaml
# neptune_config = read_yaml('neptune_config.yaml')

neptune_context = neptune.Context()

X_COLUMNS = ['comment_text']
Y_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': neptune_context.params.experiment_dir},
    'fill_na': {'na_columns': X_COLUMNS},
    'xy_split': {'x_columns': X_COLUMNS,
                 'y_columns': Y_COLUMNS
                 },
    'char_tokenizer': {'char_level': True,
                       'maxlen': neptune_context.params.maxlen_char,
                       'num_words': neptune_context.params.max_features_char
                       },
    'word_tokenizer': {'char_level': False,
                       'maxlen': neptune_context.params.maxlen_words,
                       'num_words': neptune_context.params.max_features_word
                       },
    'tfidf_char_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'char',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, neptune_context.params.char_ngram_max),
                              'max_features': neptune_context.params.max_features_char
                              },
    'tfidf_word_vectorizer': {'sublinear_tf': True,
                              'strip_accents': 'unicode',
                              'analyzer': 'word',
                              'token_pattern': r'\w{1,}',
                              'ngram_range': (1, 1),
                              'max_features': neptune_context.params.max_features_word
                              },
    'glove_embeddings': {'pretrained_filepath': neptune_context.params.embedding_filepath,
                         'max_features': neptune_context.params.max_features_word,
                         'embedding_size': neptune_context.params.word_embedding_size
                         },
    'glove_dpcnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_word,
                                                 'maxlen': neptune_context.params.maxlen_words,
                                                 'embedding_size': neptune_context.params.word_embedding_size,
                                                 'trainable_embedding': neptune_context.params.trainable_embedding,
                                                 'filter_nr': neptune_context.params.dpcnn_filter_nr,
                                                 'kernel_size': neptune_context.params.dpcnn_kernel_size,
                                                 'repeat_block': neptune_context.params.dpcnn_repeat_block,
                                                 'dense_size': neptune_context.params.dpcnn_dense_size,
                                                 'repeat_dense': neptune_context.params.dpcnn_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'use_batch_norm': neptune_context.params.use_batch_norm,
                                                 'dropout_convo': neptune_context.params.dropout_convo,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     'momentum': neptune_context.params.momentum,
                                                     'nesterov': True
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'shuffle': True,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'glove_dpcnn_network',
                                     'glove_dpcnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'glove_scnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_word,
                                                 'maxlen': neptune_context.params.maxlen_words,
                                                 'embedding_size': neptune_context.params.word_embedding_size,
                                                 'trainable_embedding': neptune_context.params.trainable_embedding,
                                                 'filter_nr': neptune_context.params.scnn_filter_nr,
                                                 'kernel_size': neptune_context.params.scnn_kernel_size,
                                                 'dense_size': neptune_context.params.scnn_dense_size,
                                                 'repeat_dense': neptune_context.params.scnn_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'use_batch_norm': neptune_context.params.use_batch_norm,
                                                 'dropout_convo': neptune_context.params.dropout_convo,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     'momentum': neptune_context.params.momentum,
                                                     'nesterov': True
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'shuffle': True,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'glove_scnn_network',
                                     'glove_scnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'glove_lstm_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_word,
                                                 'maxlen': neptune_context.params.maxlen_words,
                                                 'embedding_size': neptune_context.params.word_embedding_size,
                                                 'trainable_embedding': neptune_context.params.trainable_embedding,
                                                 'unit_nr': neptune_context.params.lstm_unit_nr,
                                                 'repeat_block': neptune_context.params.lstm_repeat_block,
                                                 'global_pooling': neptune_context.params.global_pooling,
                                                 'dense_size': neptune_context.params.lstm_dense_size,
                                                 'repeat_dense': neptune_context.params.lstm_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'dropout_lstm': neptune_context.params.dropout_lstm,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'glove_lstm_network',
                                     'glove_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'word_lstm_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_word,
                                                 'maxlen': neptune_context.params.maxlen_words,
                                                 'embedding_size': neptune_context.params.word_embedding_size,
                                                 'trainable_embedding': neptune_context.params.trainable_embedding,
                                                 'unit_nr': neptune_context.params.lstm_unit_nr,
                                                 'repeat_block': neptune_context.params.lstm_repeat_block,
                                                 'global_pooling': neptune_context.params.global_pooling,
                                                 'dense_size': neptune_context.params.lstm_dense_size,
                                                 'repeat_dense': neptune_context.params.lstm_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'dropout_lstm': neptune_context.params.dropout_lstm,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints', 'word_lstm_network',
                                     'word_lstm_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'word_dpcnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_word,
                                                 'maxlen': neptune_context.params.maxlen_words,
                                                 'embedding_size': neptune_context.params.word_embedding_size,
                                                 'trainable_embedding': neptune_context.params.trainable_embedding,
                                                 'filter_nr': neptune_context.params.dpcnn_filter_nr,
                                                 'kernel_size': neptune_context.params.dpcnn_kernel_size,
                                                 'repeat_block': neptune_context.params.dpcnn_repeat_block,
                                                 'dense_size': neptune_context.params.dpcnn_dense_size,
                                                 'repeat_dense': neptune_context.params.dpcnn_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'use_batch_norm': neptune_context.params.use_batch_norm,
                                                 'dropout_convo': neptune_context.params.dropout_convo,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     'momentum': neptune_context.params.momentum,
                                                     'nesterov': True
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'shuffle': True,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'word_dpcnn_network',
                                     'word_dpcnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'char_cnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_char,
                                                 'maxlen': neptune_context.params.maxlen_char,
                                                 'embedding_size': neptune_context.params.char_embedding_size,
                                                 'filter_nr': neptune_context.params.char_cnn_filter_nr,
                                                 'kernel_size': neptune_context.params.char_cnn_kernel_size,
                                                 'dense_size': neptune_context.params.char_cnn_dense_size,
                                                 'repeat_dense': neptune_context.params.char_cnn_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'use_batch_norm': neptune_context.params.use_batch_norm,
                                                 'dropout_convo': neptune_context.params.dropout_convo,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'char_cnn_network',
                                     'char_cnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'char_vdcnn_network': {
        'architecture_config': {'model_params': {'max_features': neptune_context.params.max_features_char,
                                                 'maxlen': neptune_context.params.maxlen_char,
                                                 'embedding_size': neptune_context.params.char_embedding_size,
                                                 'filter_nr': neptune_context.params.char_cnn_filter_nr,
                                                 'kernel_size': neptune_context.params.char_vdcnn_kernel_size,
                                                 'repeat_block': neptune_context.params.char_vdcnn_repeat_block,
                                                 'dense_size': neptune_context.params.char_vdcnn_dense_size,
                                                 'repeat_dense': neptune_context.params.char_vdcnn_repeat_dense,
                                                 'l2_reg': neptune_context.params.l2_reg_convo,
                                                 'use_prelu': neptune_context.params.use_prelu,
                                                 'use_batch_norm': neptune_context.params.use_batch_norm,
                                                 'dropout_convo': neptune_context.params.dropout_convo,
                                                 'dropout_dense': neptune_context.params.dropout_dense
                                                 },
                                'optimizer_params': {'lr': neptune_context.params.lr,
                                                     },
                                },
        'training_config': {'epochs': neptune_context.params.epochs_nr,
                            'batch_size': neptune_context.params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(neptune_context.params.experiment_dir, 'checkpoints',
                                     'char_cnn_network',
                                     'char_cnn_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': neptune_context.params.gamma},
            'early_stopping': {'patience': neptune_context.params.patience},
            'neptune_monitor': {},
        },
    },
    'logistic_regression_multilabel': {'label_nr': 6,
                                       'C': neptune_context.params.log_reg_c,
                                       'solver': 'sag',
                                       'n_jobs': neptune_context.params.num_workers,
                                       },
    'logistic_regression_ensemble': {'label_nr': 6,
                                     'C': neptune_context.params.ensemble_log_reg_c,
                                     'n_jobs': neptune_context.params.num_workers,
                                     },
    'prediction_average': {'weights': None
                           }
})
