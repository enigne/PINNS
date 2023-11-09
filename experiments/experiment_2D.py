import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *
from equations import *

def experiment_2D_3NN_hyperparameter_search(weights, epochADAM=100000, epochLBFGS=50000, N_u=50, N_f=100, seed=1234, log_frequency=1000, history_frequency=10, NLayers=8, NNeurons=20, noiseLevel=[]): #{{{
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
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 1-sized output [u, v]
    hp["layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 2-sized output [h, H]
    hp["h_layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 1-sized output [C]
    hp["C_layers"] = [2]+[NNeurons]*NLayers+[1]
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
        modelPath = "./Models/SSA2D_3NN_"+str(NLayers)+"x"+str(NNeurons)+"_noise_" + "".join([str(i)+"_" for i in noiseLevel])+ "weights" + "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    else:
        modelPath = "./Models/SSA2D_3NN_"+str(NLayers)+"x"+str(NNeurons)+"_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
        
    modelPath += ("_seed_" + str(seed) if seed else "")
   # + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_all(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA2D_3NN_calvingfront_invertC(hp, logger, X_f,
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
    plot_2D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
    #}}}
def experiment_2D_vel_hyperparameter_search(weights, epochADAM=100000, epochLBFGS=50000, N_u=50, N_f=100, seed=1234, log_frequency=1000, history_frequency=10, NLayers=8, NNeurons=20, noiseLevel=[]): #{{{
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
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 1-sized output [u, v]
    hp["layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 2-sized output [h, H]
    hp["h_layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x], NLayers hidden layer of NNeurons-width, 1-sized output [C]
    hp["C_layers"] = [2]+[NNeurons]*NLayers+[1]
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
        modelPath = "./Models/SSA2D_3NN_vel_"+str(NLayers)+"x"+str(NNeurons)+"_noise_" + "".join([str(i)+"_" for i in noiseLevel])+ "weights" + "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    else:
        modelPath = "./Models/SSA2D_3NN_vel_"+str(NLayers)+"x"+str(NNeurons)+"_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
        
    modelPath += ("_seed_" + str(seed) if seed else "")
   # + "ADAM"+str(hp["tf_epochs"]) +"_BFGS"+str(hp["nt_epochs"])
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_all(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA2D_3NN_solve_vel(hp, logger, X_f,
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
    plot_2D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
    #}}}
def experiment_2D_frictionNN_hyperparameter_search(weights, epochADAM=400000, epochLBFGS=0, N_u=2000, N_f=4000, seed=1234, log_frequency=10000, history_frequency=10, NLayers=6, NNeurons=20,  #{{{
                                                    inputFileName="Helheim_Weertman_iT080_PINN_fastflow_CF", outputFileName="SSA2D_frictionuvsH"):
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
    # DeepNN topology (2-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [u,v]
    hp["layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 2-sized output [s, H]
    hp["h_layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [C]
    hp["C_layers"] = [2]+[NNeurons]*NLayers+[1]
    # DeepNN topology (4-sized input [u,v,s,H], NLayers hidden layer of NNeurons-width, 1-sized output [taub]
    hp["friction_layers"] = [4]+[NNeurons]*NLayers+[1]
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
    path = os.path.join(appDataPath, inputFileName)
    
    # create output folder
    loss_weights = [10**(-w) for w in weights]
    now = datetime.now()
    modelPath = "./Models/"+outputFileName+"_3NN_"+str(NLayers)+"x"+str(NNeurons)+"_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    modelPath += ("_seed_" + str(seed) if seed else "")
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_all(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA2D_frictionNN_uvsH(hp, logger, X_f,
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
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot
    plot_2D_frictionNN(pinn, X_f, X_star, u_star, xlb, xub, 
                          vranges={'u - u obs': [-1e3,1e3], 'v - v obs': [-1e3,1e3], 
                                   'h - h obs': [-1e2,1e2], 'H - H obs': [-1e2,1e2], 
                                   'C - C obs': [-1e3,1e3], 'taub pred': [0, 1e6], 
                                   'taub from ISSM C': [0,1e6], 'taub - taub obs': [-1e6,1e6]},
                                    savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
#}}}
def experiment_2D_frictionNN_positivetau(weights, epochADAM=400000, epochLBFGS=0, N_u=2000, N_f=4000, seed=1234, log_frequency=10000, history_frequency=10, NLayers=6, NNeurons=20,  #{{{
                                                    inputFileName="Helheim_Weertman_iT080_PINN_fastflow_CF", outputFileName="SSA2D_frictionuvsH"):
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
    # DeepNN topology (2-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [u,v]
    hp["layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 2-sized output [s, H]
    hp["h_layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [C]
    hp["C_layers"] = [2]+[NNeurons]*NLayers+[1]
    # DeepNN topology (4-sized input [u,v,s,H], NLayers hidden layer of NNeurons-width, 1-sized output [taub]
    hp["friction_layers"] = [4]+[NNeurons]*NLayers+[1]
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
    path = os.path.join(appDataPath, inputFileName)
    
    # create output folder
    loss_weights = [10**(-w) for w in weights]
    now = datetime.now()
    modelPath = "./Models/"+outputFileName+"_3NN_"+str(NLayers)+"x"+str(NNeurons)+"_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    modelPath += ("_seed_" + str(seed) if seed else "")
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_all(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA2D_frictionNN_uvsH_positiveTau(hp, logger, X_f,
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
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot
    plot_2D_frictionNN(pinn, X_f, X_star, u_star, xlb, xub, 
                          vranges={'u - u obs': [-1e3,1e3], 'v - v obs': [-1e3,1e3], 
                                   'h - h obs': [-1e2,1e2], 'H - H obs': [-1e2,1e2], 
                                   'C - C obs': [-1e3,1e3], 'taub pred': [0, 1e6], 
                                   'taub from ISSM C': [0,1e6], 'taub - taub obs': [-1e6,1e6]},
                                    savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
#}}}
def experiment_2D_frictionNN_positivetau_velmag(weights, epochADAM=400000, epochLBFGS=0, N_u=2000, N_f=4000, seed=1234, log_frequency=10000, history_frequency=10, NLayers=6, NNeurons=20,  #{{{
                                                    inputFileName="Helheim_Weertman_iT080_PINN_fastflow_CF", outputFileName="SSA2D_frictionuvsH"):
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
    # DeepNN topology (2-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [u,v]
    hp["layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 2-sized output [s, H]
    hp["h_layers"] = [2]+[NNeurons]*NLayers+[2]
    # DeepNN topology (1-sized input [x,y], NLayers hidden layer of NNeurons-width, 1-sized output [C]
    hp["C_layers"] = [2]+[NNeurons]*NLayers+[1]
    # DeepNN topology (4-sized input [velmag,s,H], NLayers hidden layer of NNeurons-width, 1-sized output [taub]
    hp["friction_layers"] = [3]+[NNeurons]*NLayers+[1]
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
    path = os.path.join(appDataPath, inputFileName)
    
    # create output folder
    loss_weights = [10**(-w) for w in weights]
    now = datetime.now()
    modelPath = "./Models/"+outputFileName+"_3NN_"+str(NLayers)+"x"+str(NNeurons)+"_weights"+ "".join([str(w)+"_" for w in weights]) + now.strftime("%Y%m%d_%H%M%S")
    modelPath += ("_seed_" + str(seed) if seed else "")
    reloadModel = False # reload from previous training
    #}}}
    # load the data {{{
    x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu = prep_Helheim_data_all(path, hp["N_u"], hp["N_f"]) #}}}
    # Creating the model and training {{{
    logger = Logger(hp)
    pinn = SSA2D_frictionNN_uvsH_positiveTau_velmag(hp, logger, X_f,
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
    # train the model {{{
    pinn.fit(X_u_train, u_train)
    # }}}
    # save {{{
    pinn.save()
    # plot
    plot_2D_frictionNN(pinn, X_f, X_star, u_star, xlb, xub, 
                          vranges={'u - u obs': [-1e3,1e3], 'v - v obs': [-1e3,1e3], 
                                   'h - h obs': [-1e2,1e2], 'H - H obs': [-1e2,1e2], 
                                   'C - C obs': [-1e3,1e3], 'taub pred': [0, 1e6], 
                                   'taub from ISSM C': [0,1e6], 'taub - taub obs': [-1e6,1e6]},
                                    savePath=modelPath)
    # history
    plot_log_history(pinn, modelPath)
    #}}}
#}}}
