import scipy.io
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from pyDOE import lhs
import os
import sys
from scipy.interpolate import griddata

# "." for Colab/VSCode, and ".." for GitHub
repoPath = "."
appDataPath = os.path.join(repoPath, "SSA", "DATA")

def prep_data(path, N_u=None, N_f=None, N_n=None, q=None, ub=None, lb=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    
    # real() is to make it float by default, in case of zeroes
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_b = np.real(data['b'].flatten()[:,None])
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    #u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))

    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None])) #, Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))

    # Domain bounds: for regularization and generate training set
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0) 
    
    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
    X_u_train = np.vstack([X_bc])
    u_train = np.vstack([u_bc])

    # Generating the x and t collocation points for f, with each having a N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f_train = lb + (ub-lb)*lhs(2, N_f)

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f_train, ub, lb  #}}}
class Logger(object): #{{{
    def __init__(self, frequency=10):
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.frequency = frequency

    def __get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def __get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")
        self.model = model
        print(self.model.summary())

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  error = {self.__get_error_u():.4e}  " + custom)

    def log_train_opt(self, name):
        # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  error = {self.__get_error_u():.4e}  " + custom)
# }}}
