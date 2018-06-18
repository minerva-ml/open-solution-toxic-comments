from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from steps.keras.callbacks import NeptuneMonitor, ReduceLR, UnfreezeLayers
from steps.keras.models import CharVDCNNTransformer, WordCuDNNGRUTransformer, WordCuDNNLSTMTransformer, \
    WordDPCNNTransformer, WordSCNNTransformer
from steps.utils import create_filepath


def create_callbacks(**kwargs):
    lr_scheduler = ReduceLR(**kwargs['lr_scheduler'])
    unfreeze_layers = UnfreezeLayers(**kwargs['unfreeze_layers'])
    early_stopping = EarlyStopping(**kwargs['early_stopping'])
    checkpoint_filepath = kwargs['model_checkpoint']['filepath']
    create_filepath(checkpoint_filepath)
    model_checkpoint = ModelCheckpoint(**kwargs['model_checkpoint'])
    neptune = NeptuneMonitor(**kwargs['neptune_monitor'])
    return [neptune, lr_scheduler, unfreeze_layers, early_stopping, model_checkpoint]


class CharVDCNN(CharVDCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return create_callbacks(**kwargs)


class WordSCNN(WordSCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return create_callbacks(**kwargs)


class WordDPCNN(WordDPCNNTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return create_callbacks(**kwargs)


class WordCuDNNGRU(WordCuDNNGRUTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return create_callbacks(**kwargs)


class WordCuDNNLSTM(WordCuDNNLSTMTransformer):
    def _build_optimizer(self, **kwargs):
        return Adam(lr=kwargs['lr'])

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        return create_callbacks(**kwargs)
