import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from equations import *

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

loss_weights=[1e-1, 1e-2, 1e-2, 1e-4, 1e-14]
# Hyper parameters {{{
hp = {}
# Data size on the solution u
hp["N_u"] = 50
# Collocation points size, where we’ll check for f = 0
hp["N_f"] = 100
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
hp["layers"] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
hp["h_layers"] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 2]
hp["C_layers"] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 1000
hp["tf_lr"] = 0.001
hp["tf_b1"] = 0.99
hp["tf_eps"] = 1e-1
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 500
hp["log_frequency"] = 100
# Record the history
hp["save_history"] = True
hp["history_frequency"] = 10
# path for loading data and saving models
repoPath = "./"
appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
path = os.path.join(appDataPath, "Helheim_Weertman_iT080_PINN_flowline_CF.mat")

modelPath = "./Models/SSA1D_weights"+ "".join([str(round(abs(math.log(w, 10))))+"_" for w in loss_weights]) + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
reloadModel = False # reload from previous training
#}}}

# load the data
x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_flowline(path, hp["N_u"], hp["N_f"])

# Creating the model and training
logger = Logger(hp)
pinn = SSA1D_3NN_calvingfront_invertC(hp, logger, X_f,
        X_bc, u_bc,
        X_cf, n_cf,
        xub, xlb, uub, ulb,
        modelPath, reloadModel,
        mu=mu,
        loss_weights=loss_weights)

X_u = pinn.tensor(X_star)
u = pinn.tensor(u_star)
# error function for logger
def error():
    return pinn.test_error(X_u, u)
logger.set_error_fn(error)

# train the model
pinn.fit(X_u_train, u_train)
# pinn.fit(X_bc, u_bc)

# save
pinn.save()

# plot 
plot_1D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
# history
plot_log_history(pinn, modelPath)

