import re

from deepsense import neptune
from keras import backend as K
from keras.callbacks import Callback


class NeptuneMonitor(Callback):
    def __init__(self, **kwargs):
        self.ctx = neptune.Context()
        self.batch_loss_channel_name = get_correct_channel_name(self.ctx, 'Batch Log-loss training')
        self.epoch_loss_channel_name = get_correct_channel_name(self.ctx, 'Log-loss training')
        self.epoch_val_loss_channel_name = get_correct_channel_name(self.ctx, 'Log-loss validation')

        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        self.ctx.channel_send(self.batch_loss_channel_name, self.batch_id, logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1
        self.ctx.channel_send(self.epoch_loss_channel_name, self.epoch_id, logs['loss'])
        self.ctx.channel_send(self.epoch_val_loss_channel_name, self.epoch_id, logs['loss'])


class ReduceLR(Callback):
    def __init__(self, gamma):
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs={}):
        if self.gamma is not None:
            K.set_value(self.model.optimizer.lr, self.gamma * K.get_value(self.model.optimizer.lr))


def get_correct_channel_name(ctx, name):
    channels_with_name = [channel for channel in ctx.job._channels if name in channel.name]
    if len(channels_with_name) == 0:
        return name
    else:
        channel_ids = [re.split('[^\d]', channel.name)[-1] for channel in channels_with_name]
        channel_ids = sorted([int(idx) if idx != '' else 0 for idx in channel_ids])
        last_id = channel_ids[-1]
        corrected_name = '{} {}'.format(name, last_id + 1)
        return corrected_name
