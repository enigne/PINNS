import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *
from equations import *

def experiment_2D_hyperparameter_search(weights, epochADAM=100000, epochLBFGS=50000, N_u=50, N_f=100, seed=1234, log_frequency=1000, history_frequency=10): #{{{
    # Manually making sure the numpy random seeds are "the same" on all devices {{{
    if seed:
        np.random.seed(seed)
        tf.random.set_seed(seed) #}}}
    # Hyper parameters {{{
    hp = {}
    # Data size on the solution u
    hp["N_u"] = N_u
    # Collocation points size, where we’ll check for f = 0
    hp["N_f"] = N_f
    # DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
    hp["layers"] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 4]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = epochADAM
    hp["tf_lr"] = 0.001
    hp["tf_b1"] = 0.99
    hp["tf_eps"] = 1e-1
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = epochLBFGS
    hp["log_frequency"] = log_frequency
    # Record the history
    hp["save_history"] = True
    hp["history_frequency"] = history_frequency
    # path for loading data and saving models
    repoPath = "./"
    appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
    path = os.path.join(appDataPath, "Helheim_Weertman_iT080_PINN_flowline_CF_2dInv.mat")
    
    loss_weights = [10**(-w) for w in weights]

    now = datetime.now()
    modelPath = "./Models/SSA1D_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    modelPath += ("_seed_" + str(seed) if seed else "")
   # + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_flowline(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA1D_calvingfront_invertC(hp, logger, X_f,
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
    # }}}
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot 
    plot_1D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
    #}}}
def experiment_2D_3NN_hyperparameter_search(weights, epochADAM=100000, epochLBFGS=50000, N_u=50, N_f=100, seed=1234, log_frequency=1000, history_frequency=10, NLayers=8, noiseLevel=[]): #{{{
    # Manually making sure the numpy random seeds are "the same" on all devices {{{
    if seed:
        np.random.seed(seed)
        tf.random.set_seed(seed) #}}}
    # Hyper parameters {{{
    hp = {}
    # Data size on the solution u
    hp["N_u"] = N_u
    # Collocation points size, where we’ll check for f = 0
    hp["N_f"] = N_f
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 1-sized output [u]
    hp["layers"] = [2]+[20]*NLayers+[1]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 2-sized output [h, H]
    hp["h_layers"] = [2]+[20]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 1-sized output [C]
    hp["C_layers"] = [2]+[20]*NLayers+[1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = epochADAM
    hp["tf_lr"] = 0.001
    hp["tf_b1"] = 0.99
    hp["tf_eps"] = 1e-1
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = epochLBFGS
    hp["log_frequency"] = log_frequency
    # Record the history
    hp["save_history"] = True
    hp["history_frequency"] = history_frequency
    # path for loading data and saving models
    repoPath = "./"
    appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
    path = os.path.join(appDataPath, "Helheim_Weertman_iT080_PINN_fastflow_CF.mat")
    
    loss_weights = [10**(-w) for w in weights]

    now = datetime.now()

    # check the input
    if type(noiseLevel) != list:
        noiseLevel = [] # set to no noise

    if noiseLevel:
        modelPath = "./Models/SSA2D_3NN_"+str(NLayers)+"x20_noise_" + "".join([str(i)+"_" for i in noiseLevel])+ "weights" + "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    else:
        modelPath = "./Models/SSA2D_3NN_"+str(NLayers)+"x20_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
        
    modelPath += ("_seed_" + str(seed) if seed else "")
   # + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_flowline(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA1D_3NN_calvingfront_invertC(hp, logger, X_f,
            X_bc, u_bc,
            X_cf, n_cf,
            xub, xlb, uub, ulb,
            modelPath, reloadModel,
            mu=mu,
            loss_weights=loss_weights)
    
    # error function for logger
    X_u = pinn.tensor(X_star)
    u = pinn.tensor(u_star)
    def error():
        return pinn.test_error(X_u, u)
    logger.set_error_fn(error)
    # }}}
    # Add noise to the obs data {{{
    if noiseLevel:
        ns = tf.random.uniform(u_train.shape, dtype=tf.float64)
        noise = 1.0 + noiseLevel * (1.0-2.0*ns)
        u_train = noise * u_train
    # }}}
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot 
    plot_1D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
    #}}}
def experiment_2D_forward_3NN_hyperparameter_search(weights, epochADAM=100000, epochLBFGS=50000, N_u=50, N_f=100, seed=1234, log_frequency=1000, history_frequency=10, NLayers=8, noiseLevel=[]): #{{{
    # Manually making sure the numpy random seeds are "the same" on all devices {{{
    if seed:
        np.random.seed(seed)
        tf.random.set_seed(seed) #}}}
    # Hyper parameters {{{
    hp = {}
    # Data size on the solution u
    hp["N_u"] = N_u
    # Collocation points size, where we’ll check for f = 0
    hp["N_f"] = N_f
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 1-sized output [u]
    hp["layers"] = [1]+[20]*NLayers+[1]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 2-sized output [h, H]
    hp["h_layers"] = [1]+[20]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of 20-width, 1-sized output [C]
    hp["C_layers"] = [1]+[20]*NLayers+[1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = epochADAM
    hp["tf_lr"] = 0.001
    hp["tf_b1"] = 0.99
    hp["tf_eps"] = 1e-1
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = epochLBFGS
    hp["log_frequency"] = log_frequency
    # Record the history
    hp["save_history"] = True
    hp["history_frequency"] = history_frequency
    # path for loading data and saving models
    repoPath = "./"
    appDataPath = os.path.join(repoPath, "matlab_SSA", "DATA")
    path = os.path.join(appDataPath, "Helheim_Weertman_iT080_PINN_flowline_CF_2dInv.mat")
    
    loss_weights = [10**(-w) for w in weights]

    now = datetime.now()

    # check the input
    if type(noiseLevel) != list:
        noiseLevel = [] # set to no noise

    if noiseLevel:
        modelPath = "./Models/SSA1D_forward_3NN_"+str(NLayers)+"x20_noise_" + "".join([str(i)+"_" for i in noiseLevel])+ "weights" + "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    else:
        modelPath = "./Models/SSA1D_forward_3NN_"+str(NLayers)+"x20_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
        
    modelPath += ("_seed_" + str(seed) if seed else "")
   # + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_flowline(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA1D_3NN_solve_vel(hp, logger, X_f,
            X_bc, u_bc,
            X_cf, n_cf,
            xub, xlb, uub, ulb,
            modelPath, reloadModel,
            mu=mu,
            loss_weights=loss_weights)
    
    # error function for logger
    X_u = pinn.tensor(X_star)
    u = pinn.tensor(u_star)
    def error():
        return pinn.test_error(X_u, u)
    logger.set_error_fn(error)
    # }}}
    # Add noise to the obs data {{{
    if noiseLevel:
        ns = tf.random.uniform(u_train.shape, dtype=tf.float64)
        noise = 1.0 + noiseLevel * (1.0-2.0*ns)
        u_train = noise * u_train
    # }}}
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot 
    plot_1D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
    #}}}
