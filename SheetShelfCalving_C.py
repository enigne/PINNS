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
hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 10000
hp["tf_lr"] = 0.001
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
modelPath = "./Models/SheetCircleF_C"
reloadModel = False # reload from previous training
#}}}
class FrictionCDNN(NeuralNetwork): #{{{
    def __init__(self, hp, logger, X_f, xub, xlb, uub, ulb, modelPath, reloadModel=False):
        super().__init__(hp, logger, xub, xlb, uub, ulb, modelPath, reloadModel)

        # scaling factors
        self.ub = xub
        self.lb = xlb

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

    @tf.function
    def loss(self, C, C_pred):
        C0 = C[:, None]
        C0_pred = C_pred[:, None]

        mse_C = tf.reduce_mean(tf.square(C0 - C0_pred))

        return mse_C

    def predict(self, X_star):
        sol_pred = self.model(X_star)
        C_pred = sol_pred[:, None]
        return C_pred.numpy()
    #}}}
# training {{{
# set the path
x, y, X_star, u_star, X_f, xub, xlb, uub, ulb = prep_Helheim_C(path)
# Creating the model and training
logger = Logger(hp)
pinn = FrictionCDNN(hp, logger, X_f, xub, xlb, uub, ulb, modelPath, reloadModel)

# error function for logger
def error():
    u_pred = pinn.predict(X_star)
    return (np.linalg.norm(u_star[:,None] - u_pred)) / np.linalg.norm(u_star[:,None])
logger.set_error_fn(error)

# fit the data
pinn.fit(X_star, u_star)

# save the weights
pinn.save()

# plot
plot_C_train(pinn, X_star, u_star, xlb, xub)

## test load
#pinn2 = FrictionCDNN(hp, logger, X_f, xub, xlb, uub, ulb)
#pinn2.model = tf.keras.models.load_model('./Models/SheetShelf_C/')
#
## plot
#plot_C_train(pinn2, X_star, u_star, xlb, xub)
#

