import random

from deepsense import neptune
from keras import backend as K
from keras.callbacks import Callback


class NeptuneMonitor(Callback):
    def __init__(self, multi_run):
        self.ctx = neptune.Context()
        self.multi_run = multi_run
        self.suffix = self._get_suffix()
        self.epoch_id = 0
        self.batch_id = 0

    def _get_suffix(self):
        if self.multi_run:
            suffix = str(random.getrandbits(8))
        else:
            suffix = ''
        return suffix

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1

        self.ctx.channel_send('Batch Log-loss training {}'.format(self.suffix), self.batch_id, logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        self.ctx.channel_send('Log-loss training {}'.format(self.suffix), self.epoch_id, logs['loss'])
        self.ctx.channel_send('Log-loss validation {}'.format(self.suffix), self.epoch_id, logs['val_loss'])


class ReduceLR(Callback):
    def __init__(self, gamma):
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs={}):
        if self.gamma is not None:
            K.set_value(self.model.optimizer.lr, self.gamma * K.get_value(self.model.optimizer.lr))
