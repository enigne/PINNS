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
hp["N_f"] = 3000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 10000
hp["tf_lr"] = 0.01
hp["tf_b1"] = 0.99
hp["tf_eps"] = 1e-1
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 0
hp["nt_lr"] = 1.2
hp["nt_ncorr"] = 50
hp["log_frequency"] = 10
hp["use_tfp"] = False
# Record the history
hp["save_history"] = True
# path for loading data and saving models
repoPath = "/totten_1/chenggong/PINNs/"
appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
path = os.path.join(appDataPath, "SSA2D_circleF.mat")
modelPath = "./Models/SheetCircleF_H_bed"
reloadModel = False # reload from previous training
#}}}
class HBedDNN(NeuralNetwork): #{{{
    def __init__(self, hp, logger, X_f, xub, xlb, uub, ulb, modelPath="./", reloadModel=False):
        super().__init__(hp, logger, xub, xlb, uub, ulb, modelPath, reloadModel)

        # scaling factors
        self.ub = xub
        self.lb = xlb

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

    @tf.function
    def gradient_model(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            Xtemp = tf.concat([self.x_f, self.y_f], axis=1)

            Hb = self.model(Xtemp)
            H = Hb[:, 0:1]
            b = Hb[:, 1:2]
            h = H + b

        h_x = tape.gradient(h, self.x_f)
        h_y = tape.gradient(h, self.y_f)
        del tape

        return h_x, h_y

    @tf.function
    def loss(self, hb, hb_pred):
        h0 = hb[:, 0:1]
        b0 = hb[:, 1:2]

        h0_pred = hb_pred[:, 0:1]
        b0_pred = hb_pred[:, 1:2]
        h_x, h_y = self.gradient_model()

        mse_h = tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_b = tf.reduce_mean(tf.square(b0 - b0_pred))
        mse_hx = 0*1e8*tf.reduce_mean(tf.square(h_x))
        mse_hy = 0*1e8*tf.reduce_mean(tf.square(h_y))

#        tf.print(f"mse_u {mse_u}    mse_v {mse_v}    mse_f1    {mse_f1}    mse_f2    {mse_f2}")
        return mse_h+mse_b#+mse_hx+mse_hy

    def predict(self, X_star):
        h_pred = self.model(X_star)
        u_pred = h_pred[:, 0:1]
        v_pred = h_pred[:, 1:2]
        return u_pred.numpy(), v_pred.numpy()
    #}}}
# Training {{{
# set the path
x, y, X_star, u_star, X_f, xub, xlb, uub, ulb = prep_Helheim_H_bed(path)
# Creating the model and training
logger = Logger(hp)
pinn = HBedDNN(hp, logger, X_f, xub, xlb, uub[0:2], ulb[0:2], modelPath)

# error function for logger
def error():
    u_pred, v_pred = pinn.predict(X_star)
    return (np.linalg.norm(u_star[:,0:1] - u_pred, 2)+np.linalg.norm(u_star[:,1:2] - v_pred, 2)) / np.linalg.norm(u_star[:,0:2], 2)
logger.set_error_fn(error)

# fit the data
pinn.fit(X_star, u_star)

# save the weights
pinn.save()

# plot
plot_H_bed_train(pinn, X_star, u_star, xlb, xub)

## test load
#pinn2 = HBedDNN(hp, logger, X_f, xub, xlb, uub, ulb)
#pinn2.model = tf.keras.models.load_model(modelSavePath)
#
## plot
#plot_H_bed_train(pinn2, X_star, u_star, xlb, xub)
#}}}

