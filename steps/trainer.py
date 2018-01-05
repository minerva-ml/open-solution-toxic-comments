from sklearn.metrics import accuracy_score


class BasicTrainer:
    def __init__(self, pipeline, config, dev_mode=False):
        self.dev_mode = dev_mode
        self.config = config
        self.pipeline = pipeline(config)
        self.cv_splitting = None

    def train(self):
        (X_train, y_train), (X_valid, y_valid) = self._load_train_valid()
        self.pipeline.fit_transform({'input': {'X': X_train,
                                               'y': y_train,
                                               'validation_data': (X_valid, y_valid),
                                               'inference': False}})

    def evaluate(self):
        _, (X_valid, y_valid) = self._load_train_valid()
        (X_test, y_test) = self._load_test()

        score_valid = self._evaluate(X_valid, y_valid)
        score_test = self._evaluate(X_test, y_test)

        return score_valid, score_test

    def substitute(self, task_handler, solution, config):
        user_pipeline = task_handler(self.pipeline, solution, config)
        self.pipeline = user_pipeline

    def save(self):
        self.pipeline.save(self.config['global']['save_filepath'])

    def load(self):
        self.pipeline.load(self.config['global']['save_filepath'])

    def _evaluate(self, X, y):
        predictions = self.pipeline.transform({'input': {'X': X,
                                                         'y': None,
                                                         'validation_data': None,
                                                         'inference': True}})
        y_pred = predictions['y_pred']
        score = accuracy_score(y_pred, y)
        return score

    def _load_train_valid(self):
        return NotImplementedError

    def _load_test(self):
        return NotImplementedError

    def _load_grid_search_params(self):
        return NotImplementedError
