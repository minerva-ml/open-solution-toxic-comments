from deepsense import neptune
from keras import backend as K
from keras.callbacks import Callback


class NeptuneMonitor(Callback):
    def __init__(self):
        self.ctx = neptune.Context()
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1

        self.ctx.channel_send('Batch Log-loss training', self.batch_id, logs['loss'])
        self.ctx.channel_send('Batch Accuracy training', self.batch_id, logs['acc'])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        self.ctx.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        self.ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        self.ctx.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        self.ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])


class ReduceLR(Callback):
    def __init__(self, gamma):
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs={}):
        if self.gamma is not None:
            K.set_value(self.model.optimizer.lr, self.gamma * K.get_value(self.model.optimizer.lr))