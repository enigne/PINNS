import json
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from typing import Dict
import pickle

class Logger(object):
    def __init__(self, hp):
        print("Hyperparameters:")
        print(json.dumps(hp, indent=2))
        print()

        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.config.list_physical_devices('GPU')))

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.frequency = hp["log_frequency"]
        self.history_frequency = hp.setdefault('history_frequency', 1)
        self.save_history = hp.setdefault('save_history', False)
        self._history = {}
    
    @property
    def history(self) -> Dict[str, np.array]:
        return self._history

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time) \
            .strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time) \
                .strftime("%M:%S")

    def get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model, model_description=True):
        print("\nTraining started")
        print("================")
        self.model = model
        if model_description:
            print(model.summary())

    def log_train_epoch(self, epoch, loss_values, custom="", is_iter=False):
        if self.save_history:
            if epoch % self.history_frequency == 0:
                for name, loss in loss_values.items():
                    record = self._history.setdefault(name, [])
                    loss_val = loss.numpy()
                    record.append(loss_val)
                test_history = self._history.setdefault("test", [])
                test_history.append(self.get_error_u())

        if epoch % self.frequency == 0:
            name = 'nt_epoch' if is_iter else 'tf_epoch'
            print(f"{name} = {epoch:6d}  " +
                  f"elapsed = {self.get_elapsed()} " +
                  f"(+{self.get_epoch_duration()})  " +
                  f"loss = {loss_values['loss']:.4e}  " + custom)

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " +
              f"error = {self.get_error_u():.4e}  " + custom)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._history, f, pickle.HIGHEST_PROTOCOL)

