import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class MinmaxScaleLayer(tf.keras.layers.Layer):
    '''
    class of min-max scaling layer
    '''
    def __init__(self, lb, ub, scale=2.0, offset=1.0, dtype="float64"):
        super(MinmaxScaleLayer, self).__init__()
        self.lb = tf.Variable(lb, dtype=dtype, trainable=False)
        self.ub = tf.Variable(ub, dtype=dtype, trainable=False)
        self.scale = tf.Variable(scale, dtype=dtype, trainable=False)
        self.offset = tf.Variable(offset, dtype=dtype, trainable=False)

    def call(self, inputs):
        return self.scale*(inputs - self.lb)/(self.ub - self.lb) - self.offset

class UpScaleLayer(tf.keras.layers.Layer):
    '''
    class of min-max up upscaling layer
    '''
    def __init__(self, lb, ub, scale=0.5, offset=1.0, dtype="float64"):
        super(UpScaleLayer, self).__init__()
        self.lb = tf.Variable(lb, dtype=dtype, trainable=False)
        self.ub = tf.Variable(ub, dtype=dtype, trainable=False)
        self.scale = tf.Variable(scale, dtype=dtype, trainable=False)
        self.offset = tf.Variable(offset, dtype=dtype, trainable=False)

    def call(self, inputs):
        return self.lb + self.scale*(inputs + self.offset)*(self.ub - self.lb)

def create_NN(layers, inputRange=(0.0, 1.0), outputRange=(0.0, 1.0), activation=tf.nn.tanh):
    '''
    create a NN with according to the layers, scale input and upscale the output
    '''
    # initialize a sequential model
    model = tf.keras.Sequential()
    # input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    # normalization layer
    model.add(MinmaxScaleLayer(inputRange[0], inputRange[1]))

    # hiden layers
    for width in layers[1:-1]:
        model.add(tf.keras.layers.Dense(
                width, activation=activation,
                kernel_initializer="glorot_normal"))

    # output layer
    model.add(tf.keras.layers.Dense(
            layers[-1], activation=None,
            kernel_initializer="glorot_normal"))

    # denormalization layer
    model.add(UpScaleLayer(outputRange[0], outputRange[1]))

    return model

class NeuralNetwork(object):
    '''
    main class of PINNs
    '''
    def __init__(self, hp, logger, xub, xlb, uub, ulb, modelPath, reloadModel=False):

        layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        # params for L-BFGS
        self.nt_config = Struct()
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps

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

        # use double percision
        self.dtype = "float64"

        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)

        if reloadModel and os.path.exists(self.modelPath):
            #load 
            self.model = tf.keras.models.load_model(modelPath)
        else:
            # create a new NN
            self.model = create_NN(layers, inputRange=(xlb, xub), outputRange=(ulb, uub))

        # setup trainable layers, will be used in L-BFGS
        # need to be overwritten if using more than one NN
        self.trainableLayers = self.model.layers[1:-1]
        self.trainableVariables = self.model.trainable_variables

    @tf.function
    def loss(self, u, X_u):
        '''
        Defining custom loss
        '''
        u_pred = self.model(X_u)
        return {"loss": tf.reduce_mean(tf.square(u - u_pred))}

    @tf.function
    def grad(self, X, u):
        '''
        Compute the gradient of the loss function with respect to the training variables
        '''
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, X)
        grads = tape.gradient(loss_value["loss"], self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        '''
        return all list of the trainable variables, if using multiple NNs, return a list of all the list 
        '''
        var = self.trainableVariables 
        return var

    def get_indices(self):
        ''' 
        get indices of all the trainable varialbes using tf.dynamic_stitch and tf.dynamic_partition
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
        one step in LBFGS: calculate the gradients also return the value of the loss function
        '''
        with tf.GradientTape() as tape:
            self.set_model_parameters(params, partition_indices)
            loss_value = self.loss(u, X_u)

        # compute the gradient
        grads = tape.gradient(loss_value["loss"], self.wrap_training_variables())
        grad_flat = tf.dynamic_stitch(stitch_indices, grads)

        return loss_value, grad_flat

    def LBFGS_optimization(self, X_u, u):
        '''
        L-BFGS: using tensorflow_probability
        '''
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
        '''
        Adams
        '''
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.Adam_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def Adam_optimization_step(self, X_u, u):
        '''
        one step in Adam
        '''
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def fit(self, X_u, u):
        '''
        main function to run the trainning: Adams + L-BFGS
        '''
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.Adam_optimization(X_u, u)

        # use LBFGS
        self.LBFGS_optimization(X_u, u)

        # log
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

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
    pass

class Struct(dummy):
    def __getattribute__(self, key):
        if key == '__dict__':
            return super(dummy, self).__getattribute__('__dict__')
        return self.__dict__.get(key, 0)
