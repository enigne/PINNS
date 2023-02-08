import json
import tensorflow as tf
import time
from datetime import datetime


class Logger(object):
    def __init__(self, hp):
        print("Hyperparameters:")
        print(json.dumps(hp, indent=2))
        print()

        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.frequency = hp["log_frequency"]
        
        self.save_history = hp["save_history"] if "save_history" in hp.keys() else False
        self.loss_history = []
        self.test_history = []

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

    def log_train_start(self, model, model_description=False):
        print("\nTraining started")
        print("================")
        self.model = model
        if model_description:
            print(model.summary())

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if self.save_history:
            if is_iter:
                tf.py_function(self.loss_history.append, inp=[loss], Tout=[])
            else:
                tf.py_function(self.loss_history.append, inp=[loss.numpy()], Tout=[])
            tf.py_function(self.test_history.append, inp=[self.get_error_u()], Tout=[])

        if epoch % self.frequency == 0:
            name = 'nt_epoch' if is_iter else 'tf_epoch'
            print(f"{name} = {epoch:6d}  " +
                  f"elapsed = {self.get_elapsed()} " +
                  f"(+{self.get_epoch_duration()})  " +
                  f"loss = {loss:.4e}  " + custom)

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " +
              f"error = {self.get_error_u():.4e}  " + custom)
