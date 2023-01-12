import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
sys.path.append(".")
sys.path.append("./utils")
from custom_lbfgs import *
from SSAutil import *
from neuralnetwork import NeuralNetwork
from logger import Logger
import matplotlib.pyplot as plt

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# Hyper parameters {{{
hp = {}
# Data size on the solution u
hp["N_0"] = 50
# Collocation points size, where we’ll check for f = 0
hp["N_f"] = 1000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 300
hp["tf_lr"] = 0.1
hp["tf_b1"] = 0.99
hp["tf_eps"] = 1e-1
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 2000
hp["nt_lr"] = 1.2
hp["nt_ncorr"] = 50
hp["log_frequency"] = 10
#}}}
class SSAInformedNN(NeuralNetwork): #{{{
    def __init__(self, hp, logger, X_f, xub, xlb, uub, ulb, eta, n=3.0):
        super().__init__(hp, logger, xub, xlb, uub, ulb)

        # scaling factors
        self.ub = xub
        self.lb = xlb

        # viscosity
        self.eta = eta
        self.n = n

        # some constants
        self.rhoi = 917  # kg/m^3
        self.rhow = 1023 # kg/m^3
        self.g = 9.81    # m/s^2
        self.hmin = 300
        self.hmax = 1000
        #self.C = 100
        self.C = tf.Variable([0.2], dtype=self.dtype)

        self.yts = 3600.0*24*365

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

    # set geometry as a function
    def geometry_model(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        
        # load the range
        xmax, ymax = self.ub
        xmin, ymin = self.lb
        hmax = self.hmax
        hmin = self.hmin
        
        # ice shelf 
        H = hmax + (hmin-hmax)*(y-ymin)/(ymax-ymin) + 0.1*(hmin-hmax)*(x-xmin)/(xmax-xmin)
        bed = 20 - self.rhoi/self.rhow*H
        return H, bed
    
    # Decomposes the multi-output into the x and y coordinates
    def uvx_model(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            UV = self.model(Xtemp)
            u = UV[:, 0:1]
            v = UV[:, 1:2]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y

    # The actual PINN
    def f_model(self):
        # viscosity
        eta = self.eta
        n = self.n

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and y
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            # Packing together the inputs
            X_f = tf.concat([self.x_f, self.y_f], axis=1)

            # get ice thickness and bed
            H, bed = self.geometry_model(X_f)
            h = H + bed

            # Getting the prediction
            u, v, u_x, v_x, u_y, v_y = self.uvx_model(X_f)

            epsilon = 0.5*eta *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = epsilon * H
            B11 = etaH*(4*u_x + 2*v_y)
            B22 = etaH*(4*v_y + 2*u_x)
            B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)
        sigma12 = tape.gradient(B12, self.y_f)

        sigma21 = tape.gradient(B12, self.x_f)
        sigma22 = tape.gradient(B22, self.y_f)

        # surface gradient
        h_x = tape.gradient(h, self.x_f)
        h_y = tape.gradient(h, self.y_f)

        # Letting the tape go
        del tape
        u_norm = (u**2+v**2+1.0e-30)**0.5
        alpha = (self.C*100.0)**2 * (u_norm)**(1.0/self.n - 1)

        f1 = sigma11 + sigma12 - alpha*u - self.rhoi*self.g*H*h_x
        f2 = sigma21 + sigma22 - alpha*v - self.rhoi*self.g*H*h_y

        return f1, f2

    def loss(self, uv, uv_pred):
        u0 = uv[:, 0:1]
        v0 = uv[:, 1:2]
        u0_pred = uv_pred[:, 0:1]
        v0_pred = uv_pred[:, 1:2]

        f1_pred, f2_pred = self.f_model()

        mse_u = 1e-6*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = 1e-6*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        mse_f1 = 1e-6*tf.reduce_mean(tf.square(f1_pred)) 
        mse_f2 = 1e-6*tf.reduce_mean(tf.square(f2_pred)) 

        tf.print(f"mse_u {mse_u}    mse_v {mse_v}    mse_f1    {mse_f1}    mse_f2    {mse_f2}")
        return mse_u+mse_v+mse_f1+mse_f2
    def wrap_training_variables(self):
        var = self.model.trainable_variables
        # add C
        var.extend([self.C])
        return var

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.model.layers[1:-1]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        # add C
        w.extend(self.C.numpy())
        return self.tensor(w)

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
        # add C
        self.C.assign([w[-1]])

    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                tape.watch(self.C)
                loss_value = self.loss(u, self.model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def predict(self, X_star):
        h_pred = self.model(X_star)
        u_pred = h_pred[:, 0:1]
        v_pred = h_pred[:, 1:2]
        return u_pred.numpy(), v_pred.numpy()
    #}}}


# set the path
repoPath = "."
appDataPath = os.path.join(repoPath, "SSA", "DATA")
path = os.path.join(appDataPath, "SSA2D.mat")
x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, xub, xlb, uub, ulb = prep_data(path, hp["N_0"], hp["N_f"])
# Creating the model and training
logger = Logger(hp)
pinn = SSAInformedNN(hp, logger, X_f, xub, xlb, uub, ulb, eta=1.8157e8)
# Linear viscosity
#pinn = SSAInformedNN(hp, logger, X_f, xub, xlb, uub, ulb, eta=1.0e15, n=1.0)

# error function for logger
def error():
    u_pred, v_pred = pinn.predict(X_star)
    return (np.linalg.norm(u_star[:,0:1] - u_pred, 2)+np.linalg.norm(u_star[:,1:2] - v_pred, 2)) / np.linalg.norm(u_star[:,0:2], 2)
logger.set_error_fn(error)
pinn.fit(X_star, u_star)
#pinn.fit(X_u_train, u_train)
u_pred, v_pred = pinn.predict(X_star)
