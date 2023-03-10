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
from plotting import *
import matplotlib.pyplot as plt

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# Hyper parameters {{{
hp = {}
# Data size on the solution u
hp["N_u"] = 2000
# Collocation points size, where we’ll check for f = 0
hp["N_f"] = 1000
# DeepNN topology (2-sized input [x t], 5 hidden layer of 20-width, 1-sized output [u]
hp["layers"] = [2, 20, 20, 20, 20, 20, 2]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 10000
hp["tf_lr"] = 0.001
hp["tf_b1"] = 0.99
hp["tf_eps"] = 1e-1
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 2000
hp["nt_lr"] = 0.8
hp["nt_ncorr"] = 50
hp["log_frequency"] = 10
#}}}
class HBedDNN(NeuralNetwork): #{{{
    def __init__(self, hp, logger, X_f, xub, xlb, uub, ulb):
        super().__init__(hp, logger, xub, xlb, uub, ulb)

        # scaling factors
        self.ub = xub
        self.lb = xlb

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

    # constrain surface gradient
    def surface_gradient(self, X):
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

    def loss(self, hb, hb_pred):
        h0 = hb[:, 0:1]
        b0 = hb[:, 1:2]
        h0_pred = hb_pred[:, 0:1]
        b0_pred = hb_pred[:, 1:2]

        mse_h = tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_b = tf.reduce_mean(tf.square(b0 - b0_pred))

        return mse_h+mse_b

    def predict(self, X_star):
        h_pred = self.model(X_star)
        u_pred = h_pred[:, 0:1]
        v_pred = h_pred[:, 1:2]
        return u_pred.numpy(), v_pred.numpy()
    #}}}


# set the path
repoPath = "."
appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
path = os.path.join(appDataPath, "SSA2D_nofriction.mat")
x, y, X_star, u_star, X_f, xub, xlb, uub, ulb = prep_H_bed(path, hp["N_f"])
# Creating the model and training
logger = Logger(hp)
pinn = HBedDNN(hp, logger, X_f, xub, xlb, uub, ulb)

# error function for logger
def error():
    u_pred, v_pred = pinn.predict(X_star)
    return (np.linalg.norm(u_star[:,0:1] - u_pred, 2)+np.linalg.norm(u_star[:,1:2] - v_pred, 2)) / np.linalg.norm(u_star[:,0:2], 2)
logger.set_error_fn(error)

# fit the data
pinn.fit(X_star, u_star)

# save the weights
pinn.model.save("./Models/H_bed/")

# plot
plot_H_bed_train(pinn, X_star, u_star, xlb, xub)

# test load
pinn2 = HBedDNN(hp, logger, X_f, xub, xlb, uub, ulb)
pinn2.model = tf.keras.models.load_model('./Models/H_bed/')

# plot
plot_H_bed_train(pinn2, X_star, u_star, xlb, xub)


