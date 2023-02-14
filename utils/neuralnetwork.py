import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import os

from custom_lbfgs import lbfgs, Struct

# min-max scaling layer
class MinmaxScaleLayer(tf.keras.layers.Layer):
    def __init__(self, lb, ub, scale=2.0, offset=1.0, dtype="float64"):
        super(MinmaxScaleLayer, self).__init__()
        self.lb = tf.Variable(lb, dtype=dtype, trainable=False)
        self.ub = tf.Variable(ub, dtype=dtype, trainable=False)
        self.scale = tf.Variable(scale, dtype=dtype, trainable=False)
        self.offset = tf.Variable(offset, dtype=dtype, trainable=False)

    def call(self, inputs):
        return self.scale*(inputs - self.lb)/(self.ub - self.lb) - self.offset

# min-max up upscaling layer
class UpScaleLayer(tf.keras.layers.Layer):
    def __init__(self, lb, ub, scale=0.5, offset=1.0, dtype="float64"):
        super(UpScaleLayer, self).__init__()
        self.lb = tf.Variable(lb, dtype=dtype, trainable=False)
        self.ub = tf.Variable(ub, dtype=dtype, trainable=False)
        self.scale = tf.Variable(scale, dtype=dtype, trainable=False)
        self.offset = tf.Variable(offset, dtype=dtype, trainable=False)

    def call(self, inputs):
        return self.lb + self.scale*(inputs + self.offset)*(self.ub - self.lb)

# main class of PINNs
class NeuralNetwork(object):
    def __init__(self, hp, logger, xub, xlb, uub, ulb, modelPath, reloadModel=False):

        layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        # params for L-BFGS
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        self.use_tfp = hp.setdefault("use_tfp", False)

        # params for Adam
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])

        # save the final model to the path
        self.modelPath = modelPath

        # logger
        self.logger = logger

        self.dtype = "float64"
        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)

        if reloadModel and os.path.exists(self.modelPath):
            #load 
            self.model = tf.keras.models.load_model(modelPath)
        else:
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

        # setup trainable layers
        self.trainableLayers = self.model.layers[1:-1]
        self.trainableVariables = self.model.trainable_variables

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i > 0:
                self.sizes_w.append(int(width * layers[i-1]))
                self.sizes_b.append(int(width))

    # Defining custom loss
    @tf.function
    def loss(self, u, u_pred):
        return {"loss": tf.reduce_mean(tf.square(u - u_pred))}

    @tf.function
    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.model(X))
        grads = tape.gradient(loss_value["loss"], self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.trainableVariables 
        return var

    def get_params(self, numpy=False):
        return []

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.trainableLayers:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = self.tensor(w)
        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.trainableLayers):
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

    def get_indices(self):
        ''' 
            to use tf.dynamic_stitch and tf.dynamic_partition
        '''
        shapes = tf.shape_n(self.wrap_training_variables())
        count = 0
        stitch_indices = []
        partition_indices = []
        for i, shape in enumerate(shapes):
            n = np.product(shape.numpy())
            stitch_indices.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            partition_indices.extend([i] * n)
            count += n
        partition_indices = tf.constant(partition_indices)
        return partition_indices, stitch_indices

    def set_model_parameters(self, params, partition_indices):
        '''
            update model parameters 
        '''
        shapes = tf.shape_n(self.wrap_training_variables())
        params = tf.dynamic_partition(params, partition_indices, len(shapes))
        for i, (shape, param) in enumerate(zip(shapes, params)):
            self.trainableVariables[i].assign(tf.reshape(param, shape))
    

    # L-BFGS 
    @tf.function
    def LBFGS_optimization_step(self, params, partition_indices, stitch_indices, X_u, u):
        '''
        calculate the gradients also return the value of the loss function
        '''
        with tf.GradientTape() as tape:
            self.set_model_parameters(params, partition_indices)
            loss_value = self.loss(u, self.model(X_u))

        # compute the gradient
        grads = tape.gradient(loss_value["loss"], self.wrap_training_variables())
        grad_flat = tf.dynamic_stitch(stitch_indices, grads)

        return loss_value, grad_flat

    def LBFGS_optimization(self, X_u, u):
        self.logger.log_train_opt("tfp-L-BFGS")
        # get the indices
        partition_indices, stitch_indices = self.get_indices()
        # counter
        epoch = tf.Variable(0)

        # a decorator to include saving and printing history
        def loss_and_grad(params):
            # the main objective function 
            loss_value, grads = self.LBFGS_optimization_step(params, partition_indices, stitch_indices, X_u, u)
            # save to log and output
            epoch.assign_add(1)
            self.logger.log_train_epoch(epoch.numpy(), loss_value, is_iter=True)

            return loss_value["loss"], grads

        # initial guess 
        init_params = tf.dynamic_stitch(stitch_indices, self.wrap_training_variables())
        max_nIter = tf.cast(self.nt_config.maxIter/3, dtype=tf.int32)

        results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=loss_and_grad,
                initial_position=init_params,
                max_iterations=max_nIter,
                tolerance=1e-30)

        # manually put the final solution back to the model
        self.set_model_parameters(results.position, partition_indices)

    # ADAM
    def Adam_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.Adam_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def Adam_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def nt_optimization(self, X_u, u):
        self.logger.log_train_opt("LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
        self.nt_optimization_steps(loss_and_flat_grad)

    def nt_optimization_steps(self, loss_and_flat_grad):
        lbfgs(loss_and_flat_grad,
              self.get_weights(),
              self.nt_config, Struct(), True,
              lambda epoch, loss, is_iter:
              self.logger.log_train_epoch(epoch, loss, "", is_iter))

    def tfp_nt_optimization_old(self, X_u, u):
        self.logger.log_train_opt("tfp-LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
        tfp.optimizer.lbfgs_minimize(
                loss_and_flat_grad,
                initial_position=self.get_weights(),
                num_correction_pairs=self.nt_config.nCorrection,
                max_iterations=self.nt_config.maxIter,
                max_line_search_iterations=50,
                initial_inverse_hessian_estimate=None,
                f_relative_tolerance=self.nt_config.tolFun,
                tolerance=1e-8, 
                parallel_iterations=20)

    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.Adam_optimization(X_u, u)

        # use LBFGS
        if self.use_tfp:
            self.LBFGS_optimization(X_u, u)
        else:
            self.nt_optimization(X_u, u)

        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    def predict(self, X_star):
        u_pred = self.model(X_star)
        return u_pred.numpy()

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def save(self):
        if hasattr(self, 'model'):
            self.model.save(self.modelPath)
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
