import shutil

import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
from gensim.models import KeyedVectors

from steps.base import BaseTransformer


class BasicClassifier(BaseTransformer):
    """
    Todo:
        load the best model at the end of the fit and save it
    """

    def __init__(self, architecture_config, training_config, callbacks_config):
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config

    def reset(self):
        self.model = self._build_model(**self.architecture_config)

    def _compile_model(self, model_params, optimizer_params):
        model = self._build_model(**model_params)
        optimizer = self._build_optimizer(**optimizer_params)
        loss = self._build_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc', ])
        return model

    def _create_callbacks(self, **kwargs):
        return NotImplementedError

    def _build_model(self, **kwargs):
        return NotImplementedError

    def _build_optimizer(self, **kwargs):
        return NotImplementedError

    def _build_loss(self, **kwargs):
        return NotImplementedError

    def save(self, filepath):
        checkpoint_callback = self.callbacks_config.get('model_checkpoint')
        if checkpoint_callback:
            checkpoint_filepath = checkpoint_callback['filepath']
            shutil.copyfile(checkpoint_filepath, filepath)
        else:
            self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath)
        return self


class ClassifierXY(BasicClassifier):
    def fit(self, X, y, validation_data):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        self.model.fit(X, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, X, y=None, validation_data=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class ClassifierGenerator(BasicClassifier):
    def fit(self, datagen, validation_datagen):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        train_flow, train_steps = datagen
        valid_flow, valid_steps = validation_datagen
        self.model.fit_generator(train_flow,
                                 steps_per_epoch=train_steps,
                                 validation_data=valid_flow,
                                 validation_steps=valid_steps,
                                 callbacks=self.callbacks,
                                 verbose=1,
                                 **self.training_config)
        return self

    def transform(self, datagen, validation_datagen=None):
        test_flow, test_steps = datagen
        predictions = self.model.predict_generator(test_flow, test_steps, verbose=1)
        return {'prediction_probability': predictions}


class EmbeddingsMatrix(BaseTransformer):
    def __init__(self, pretrained_filepath, max_features, embedding_size):
        self.pretrained_filepath = pretrained_filepath
        self.max_features = max_features
        self.embedding_size = embedding_size

    def fit(self, tokenizer):
        self.embedding_matrix = self._get_embedding_matrix(tokenizer)
        return self

    def transform(self, tokenizer):
        return {'embeddings_matrix': self.embedding_matrix}

    def _get_embedding_matrix(self, tokenizer):
        return NotImplementedError

    def save(self, filepath):
        joblib.dump(self.embedding_matrix, filepath)

    def load(self, filepath):
        self.embedding_matrix = joblib.load(filepath)
        return self


class GloveEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        embeddings_index = dict()
        with open(self.pretrained_filepath) as f:
            for line in f:
                # Note: use split(' ') instead of split() if you get an error.
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embedding_size))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix


class Word2VecEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        model = KeyedVectors.load_word2vec_format(self.pretrained_filepath, binary=True)

        emb_mean, emb_std = model.wv.syn0.mean(), model.wv.syn0.std()

        word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embedding_size))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            try:
                embedding_vector = model[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue
        return embedding_matrix


class FastTextEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        embeddings_index = dict()
        with open(self.pretrained_filepath) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0:
                    continue
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                if coefs.shape[0] != self.embedding_size:
                    continue
                embeddings_index[word] = coefs

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embedding_size))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix
