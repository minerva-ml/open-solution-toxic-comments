import numpy as np
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, \
    BatchNormalization,LeakyReLU, concatenate
from keras.layers.merge import add
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras import regularizers

from steps.models.keras.callbacks import NeptuneMonitor, ReduceLR
from steps.models.keras.models import ClassifierXY
from steps.utils import create_filepath


class CharacterClassifier(ClassifierXY):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        lr_scheduler = ReduceLR(**kwargs['lr_scheduler'])
        early_stopping = EarlyStopping(**kwargs['early_stopping'])
        checkpoint_filepath = kwargs['model_checkpoint']['filepath']
        create_filepath(checkpoint_filepath)
        model_checkpoint = ModelCheckpoint(**kwargs['model_checkpoint'])
        neptune = NeptuneMonitor()
        return [neptune, lr_scheduler, early_stopping, model_checkpoint]


class CharCNN(CharacterClassifier):
    def _build_model(self, max_features, maxlen, embedding_size):
        input_text = Input(shape=(maxlen,))

        x = Embedding(input_dim=max_features, output_dim=embedding_size)(input_text)
        x = Conv1D(64, kernel_size=6, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)

        predictions = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=input_text, outputs=predictions)
        return model


class WordTrainableLSTM(CharacterClassifier):
    def _build_model(self, maxlen, max_features, embedding_size):
        input_text = Input(shape=(maxlen,))

        x = Embedding(max_features, embedding_size)(input_text)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=input_text, outputs=predictions)
        return model


class GloveLSTM(CharacterClassifier):
    def fit(self, embedding_matrix, X, y, validation_data, ):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.architecture_config['model_params']['embedding_matrix'] = embedding_matrix
        self.model = self._compile_model(**self.architecture_config)

        self.model.fit(X, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, embedding_matrix, X, y=None, validation_data=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}

    def _build_model(self, maxlen, max_features, embedding_size, embedding_matrix):
        input_text = Input(shape=(maxlen,))

        x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(input_text)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=input_text, outputs=predictions)
        return model


class GloveCNN(CharacterClassifier):
    def fit(self, embedding_matrix, X, y, validation_data):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.architecture_config['model_params']['embedding_matrix'] = embedding_matrix
        self.model = self._compile_model(**self.architecture_config)

        self.model.fit(X, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, embedding_matrix, X, y=None, validation_data=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}

    def _build_model(self, maxlen, max_features, embedding_size, embedding_matrix):
        input_text = Input(shape=(maxlen,))

        x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(input_text)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=input_text, outputs=predictions)
        return model

class GloveDPCNN(GloveCNN):
    def _build_optimizer(self, **kwargs):
        return SGD(**kwargs)
    
    def _build_model(self, maxlen, max_features, embedding_size, embedding_matrix, filter_nr, kernel_size, repeat_block, l2_reg):
        """
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """

        def _base_layer(x):
            x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = LeakyReLU(alpha=0.3)(x)
            return x
        
        def _shape_matching_layer(x):
            x = Conv1D(filter_nr, kernel_size=1, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = LeakyReLU(alpha=0.3)(x)
            return x
        
        def _dpcnn_block(x):
            x = MaxPooling1D(pool_size=3, stride=2)(x)
            x_conv = _base_layer(x)
            x_conv = _base_layer(x_conv)
            x = add([x_conv, x])
            return x
        
        input_text = Input(shape=(maxlen,))

        embedding = Embedding(max_features, embedding_size, weights=[embedding_matrix], 
                              trainable=False)(input_text)
        x = _base_layer(embedding)
        x = _base_layer(x)
        if embedding_size == filter_nr:
            x = add([embedding, x])
        else:
            embedding_resized = _shape_matching_layer(embedding)
            x = add([embedding_resized, x])
        
        for _ in range(repeat_block):
            x = _dpcnn_block(x)

        x = GlobalMaxPool1D()(x)
        x = Dense(256)(x)
        predictions = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=input_text, outputs=predictions)
        return model

class GloveExp(GloveCNN):   
    def _build_model(self, maxlen, max_features, embedding_size, embedding_matrix):
        input_text = Input(shape=(maxlen,))

        x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=False)(input_text)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(6, activation="sigmoid")(x)

        model = Model(inputs=input_text, outputs=predictions)
        return model
