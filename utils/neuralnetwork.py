import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from custom_lbfgs import lbfgs, Struct

# min-max scaling layer
class MinmaxScaleLayer(tf.keras.layers.Layer):
    def __init__(self, lb, ub, scale=2.0, offset=1.0, dtype="float64"):
        super(MinmaxScaleLayer, self).__init__()
        self.lb = tf.constant(lb, dtype=dtype)
        self.ub = tf.constant(ub, dtype=dtype)
        self.scale = tf.constant(scale, dtype=dtype)
        self.offset = tf.constant(offset, dtype=dtype)

    def call(self, inputs):
        return self.scale*(inputs - self.lb)/(self.ub - self.lb) - self.offset

# min-max up upscaling layer
class UpScaleLayer(tf.keras.layers.Layer):
    def __init__(self, lb, ub, scale=0.5, offset=1.0, dtype="float64"):
        super(UpScaleLayer, self).__init__()
        self.lb = tf.convert_to_tensor(lb, dtype=dtype)
        self.ub = tf.constant(ub, dtype=dtype)
        self.scale = tf.constant(scale, dtype=dtype)
        self.offset = tf.constant(offset, dtype=dtype)

    def call(self, inputs):
        return self.lb + self.scale*(inputs + self.offset)*(self.ub - self.lb)

# main class of PINNs
class NeuralNetwork(object):
    def __init__(self, hp, logger, xub, xlb, uub, ulb):

        layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])

        self.dtype = "float64"
        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)
        self.model = tf.keras.Sequential()
        # input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # normalization layer
        self.model.add(MinmaxScaleLayer(xlb, xub))

        # NN layers
        for width in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        # output layer
        self.model.add(tf.keras.layers.Dense(
                layers[-1], activation=None,
                kernel_initializer="glorot_normal"))

        # denormalization layer
        self.model.add(UpScaleLayer(ulb, uub))

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i > 0:
                self.sizes_w.append(int(width * layers[i-1]))
                self.sizes_b.append(int(width))

        self.logger = logger

    # Defining custom loss
    @tf.function
    def loss(self, u, u_pred):
        return tf.reduce_mean(tf.square(u - u_pred))

    @tf.function
    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def get_params(self, numpy=False):
        return []

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.model.layers[1:-1]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = self.tensor(w)
        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.model.layers[1:-1]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss_value = self.loss(u, self.model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def tf_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def tf_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def nt_optimization(self, X_u, u):
        self.logger.log_train_opt("LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
        #tfp.optimizer.lbfgs_minimize(
        #        loss_and_flat_grad,
        #        initial_position=self.get_weights(),
        #        num_correction_pairs=self.nt_config.nCorrection,
        #        max_iterations=self.nt_config.maxIter,
        #        f_relative_tolerance=self.nt_config.tolFun,
        #        tolerance=self.nt_config.tolFun,
        #        parallel_iterations=6)
        self.nt_optimization_steps(loss_and_flat_grad)

    def nt_optimization_steps(self, loss_and_flat_grad):
        lbfgs(loss_and_flat_grad,
              self.get_weights(),
              self.nt_config, Struct(), True,
              lambda epoch, loss, is_iter:
              self.logger.log_train_epoch(epoch, loss, "", is_iter))

    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.tf_optimization(X_u, u)
        self.nt_optimization(X_u, u)

        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    def predict(self, X_star):
        u_pred = self.model(X_star)
        return u_pred.numpy()

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

#    def save(self, path, name, mtype='Umodel'):
#        if hasattr(self, 'model'):
#            self.model.save(path + name + mtype)
#        if hasattr(self, 'C_model'):
#            self.C_model.save(path + name + '/Cmodel')
#
#    def load(self, path, name, mtype='Umodel'):
#        if hasattr(self, 'model'):
#            self.model = tf.keras.models.load_model(path + name+'/Umodel')
#        if hasattr(self, 'C_model'):
#            self.C_model = tf.keras.models.load_model(path + name+'/Cmodel')

#    def save_weights(self, path, name):
#        self.model.save_weights(path + name+'/Umodel_weights')
#        self.C_model.save_weights(path + name+'/Cmodel_weights')
#
#    def load_weights(self, path, name):
#        self.model.load_weights(path + name+'/Umodel_weights')
#        self.C_model.load_weights(path + name+'/Cmodel_weights')
#
