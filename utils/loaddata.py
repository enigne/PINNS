import os
import sys
import scipy.io
import time

import numpy as np
import tensorflow as tf
from datetime import datetime
from pyDOE import lhs

def prep_data(path, N_u=None, N_f=None, invertC=False): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    
    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    if invertC:
        Exact_C = np.real(data['C'].flatten()[:,None])/100.0

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    if invertC:
        u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_C.flatten()[:,None] )) #, Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))
    else:
        u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None])) #, Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 
    
    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
    X_u_train = np.vstack([X_star])
    u_train = np.vstack([u_star])

    # Generating the x and t collocation points for f, with each having a N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f_train = xlb + (xub-xlb)*lhs(2, N_f)

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f_train, X_bc, u_bc, xub, xlb, uub, ulb  #}}}
def prep_H_bed(path, N_f=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    
    # real() is to make it float by default, in case of zeroes
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_b = np.real(data['b'].flatten()[:,None])

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star
    u_star = np.hstack((Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 
    
    # Generating the x and t collocation points for f, with each having a N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f = xlb + (xub-xlb)*lhs(2, N_f)

    return x, y, X_star, u_star, X_f, xub, xlb, uub, ulb  #}}}
def prep_Helheim_data(path, N_u=None, N_f=None, invertC=False): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]

    # collocation points
    X_f = np.real(data['X_f'])
    idf = np.random.choice(X_f.shape[0], N_f, replace=False)
    X_f = X_f[idf,:]

    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    if invertC:
        Exact_C = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    if invertC:
        u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_C.flatten()[:,None] )) #, Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))
    else:
        u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None])) #, Exact_H.flatten()[:,None], Exact_b.flatten()[:,None]))

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_u_train = np.vstack([X_star[iice[:,0],:]])
    u_train = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb  #}}}
def prep_Helheim_H_bed(path): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    X_f = np.real(data['X_f'])
    
    # real() is to make it float by default, in case of zeroes
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_b = np.real(data['b'].flatten()[:,None])
#    Exact_hx = np.real(data['ssx'].flatten()[:,None])
#    Exact_hy = np.real(data['ssy'].flatten()[:,None])

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star
#    u_star = np.hstack((Exact_H.flatten()[:,None], Exact_b.flatten()[:,None], Exact_hx.flatten()[:,None], Exact_hy.flatten()[:,None]))
    u_star = np.hstack((Exact_H.flatten()[:,None], Exact_b.flatten()[:,None])) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 
    

    return x, y, X_star, u_star, X_f, xub, xlb, uub, ulb  #}}}
def prep_Helheim_C(path): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    X_f = np.real(data['X_f'])

    # real() is to make it float by default, in case of zeroes
    Exact_C = np.real(data['C'].flatten()[:,None])

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star
    #u_star = np.hstack((Exact_C.flatten()[:,None], Exact_C.flatten()[:,None]))
    u_star = Exact_C

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 

    return x, y, X_star, u_star, X_f, xub, xlb, uub, ulb  #}}}
def prep_Helheim_data_all(path, N_u=None, N_f=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]

    # collocation points
    X_f = np.real(data['X_f'])
    idf = np.random.choice(X_f.shape[0], N_f, replace=False)
    X_f = X_f[idf,:]

    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_C = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] )) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_u_train = np.vstack([X_star[iice[:,0],:]])
    u_train = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb  #}}}
def prep_Helheim_Dirichlet(path, N_u=None, N_f=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]

    # collocation points
    X_f = np.real(data['X_f'])
    idf = np.random.choice(X_f.shape[0], N_f, replace=False)
    X_f = X_f[idf,:]

    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_C = np.real(data['C'].flatten()[:,None])

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] )) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 

    # load Dirichlet boundary conditions
    X_bc = np.real(data['X_bc'])
    u_bc = np.real(data['u_bc'])

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_u_train = np.vstack([X_star[iice[:,0],:]])
    u_train = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb  #}}}
def prep_Helheim_data_flowline(path, N_u=None, N_f=None): #{{{
    # Reading SSA ref solution along the flowline, 1D problem: x coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # viscosity
    mu = data['mu']

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]

    # collocation points
    X_f = np.real(data['X_f'])
    idf = np.random.choice(X_f.shape[0], N_f, replace=False)
    X_f = X_f[idf,:]

    # real() is to make it float by default, in case of zeroes
    Exact_vel = np.real(data['vel'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_C = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x for predictions in one single array, as X_star
    X_star = x

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vel.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] )) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],None]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_u_train = X_star[iice[:,0]]
    u_train = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    nx = data['nx'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None] )) 
    n_cf = np.hstack((nx.flatten()[:,None] ))

    return x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
