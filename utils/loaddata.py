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

    # viscosity
    mu = data['mu']

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

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
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
    X_star = X_star[iice[:,0]]
    u_star = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_u_train = X_star[idx]
    # Getting the corresponding u_train
    u_train = u_star[idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    nx = data['nx'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None] )) 
    n_cf = np.hstack((nx.flatten()[:,None] ))

    return x, Exact_vel, X_star, u_star, X_u_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
def prep_Helheim_transient(path, N_u=None, N_f=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, usol and Hsol
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # viscosity
    mu = data['mu']

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    x = data['x'].flatten()[:,None]
    t = data['t'].flatten()[:,None]

    # collocation points
    X_f = np.real(data['X_f'])
    idf = np.random.choice(X_f.shape[0], N_f, replace=False)
    X_f = X_f[idf,:]

    # real() is to make it float by default, in case of zeroes
    Exact_vel = np.real(data['vel'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_smb = np.real(data['smb'].flatten()[:,None])

    # will be used for test
    X_1d = np.real(data['x1d'].flatten()[:,None])
    C_1d = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))
    # Preparing the testing u_star 
    u_star = np.hstack((Exact_vel.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_smb.flatten()[:,None] )) 


    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    ulb = u_star.min(axis=0)
    uub = u_star.max(axis=0) 
    
    # append lb and ub for C
    ulb = np.append(ulb, C_1d.min(axis=0))
    uub = np.append(uub, C_1d.max(axis=0))

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_train = np.vstack([X_star[iice[:,0],:]])
    u_train = np.vstack([u_star[iice[:,0],:]])

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    idx = np.random.choice(X_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
    X_train = X_train[idx,:]
    # Getting the corresponding u_train
    u_train = u_train [idx,:]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    ct = data['ct'].flatten()[:,None]
    nx = data['nx'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], ct.flatten()[:,None])) 
    n_cf = np.hstack((nx.flatten()[:,None] ))

    # X_star, u_star : true solutions 
    # X_train, u_train : training data set
    # X_1d, C_1d : true static solution, C
    # X_f: colocation points
    # X_bc, u_bc : boundary nodes
    # X_cf, n_cf : cavling front positions and normal vector
    return X_star, u_star, X_train, u_train, X_1d, C_1d, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
def prep_2D_data_all(path, N_f=None, N_u=None, N_s=None, N_H=None, N_C=None, FlightTrack=False): #{{{
    # Reading SSA ref solutions: x, y-coordinates, provide ALL the variables in u_train
    data = scipy.io.loadmat(path,  mat_dtype=True)

    # viscosity
    mu = data['mu']

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
    umin = u_star.min(axis=0)
    umax = u_star.max(axis=0) 
    ulb = {}
    uub = {}
    ulb["uv"] = umin[0:2]
    uub["uv"] = umax[0:2]
    ulb["sH"] = umin[2:4]
    uub["sH"] = umax[2:4]
    ulb["C"] = umin[4:5]
    uub["C"] = umax[4:5]

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_ = np.vstack([X_star[iice[:,0],:]])
    u_ = np.vstack([u_star[iice[:,0],:]])

    # Getting the corresponding X_train and u_train(which is now scarce boundary/initial coordinates)
    X_train = {}
    u_train = {}

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    # velocity data
    if N_u:
        idx = np.random.choice(X_.shape[0], N_u, replace=False)
        X_train["uv"] = X_[idx,:]
        u_train["uv"] = u_[idx, 0:2]
    else:
        X_train["uv"] = X_bc
        u_train["uv"] = u_bc[:, 0:2]

    # surface elevation, always available, use the maximum points among all the other data set
    if N_s is None:
        Nlist = [N_u, N_H, N_C]
        N_s = max([i for i in Nlist if i is not None])

    idx = np.random.choice(X_.shape[0], N_s, replace=False)
    X_train["s"] = X_[idx,:]
    u_train["s"] = u_[idx, 2:3]

    # ice thickness, or bed elevation
    if N_H:
        if FlightTrack:
            H_ft = np.real(data['H_ft'].flatten()[:,None])
            x_ft = np.real(data['x_ft'].flatten()[:,None])
            y_ft = np.real(data['y_ft'].flatten()[:,None])
            X_ft = np.hstack((x_ft.flatten()[:,None], y_ft.flatten()[:,None]))

            N_H = min(X_ft.shape[0], N_H)
            print(f"Use {N_H} flight track data for the ice thickness training data")
            idx = np.random.choice(X_ft.shape[0], N_H, replace=False)
            X_train["H"] = np.vstack([X_bc, X_ft[idx, :]])
            u_train["H"] = np.vstack([u_bc[:, 3:4], H_ft[idx,:]])
        else:
            idx = np.random.choice(X_.shape[0], N_H, replace=False)
            X_train["H"] = X_[idx,:]
            u_train["H"] = u_[idx, 3:4]
    else:
        if 'x_fl' in data.keys():
            print('Warning, using flowlines, this should only be used for proof of concept')
            # load thickness along flowlines
            H_fl = np.real(data['H_fl'].flatten()[:,None])
            x_fl = np.real(data['x_fl'].flatten()[:,None])
            y_fl = np.real(data['y_fl'].flatten()[:,None])
            X_fl = np.hstack((x_fl.flatten()[:,None], y_fl.flatten()[:,None]))

            X_train["H"] = np.vstack([X_bc, X_fl])
            u_train["H"] = np.vstack([u_bc[:, 3:4], H_fl])
        else:
            X_train["H"] = X_bc
            u_train["H"] = u_bc[:, 3:4]

    # friction coefficients
    if N_C:
        idx = np.random.choice(X_.shape[0], N_C, replace=False)
        X_train["C"] = X_[idx,:]
        u_train["C"] = u_[idx, 4:5]
    else:
        X_train["C"] = X_bc
        u_train["C"] = u_bc[:, 4:5]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
def prep_2D_data_withmu(path, N_f=None, N_u=None, N_s=None, N_H=None, N_C=None, N_mu=None): #{{{
    # Reading SSA ref solutions: x, y-coordinates, provide ALL the variables in u_train
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
    Exact_mu = np.real(data['mu'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None], Exact_mu.flatten()[:,None])) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    umin = u_star.min(axis=0)
    umax = u_star.max(axis=0) 
    ulb = {}
    uub = {}
    ulb["uv"] = umin[0:2]
    uub["uv"] = umax[0:2]
    ulb["sH"] = umin[2:4]
    uub["sH"] = umax[2:4]
    ulb["C"] = umin[4:5]
    uub["C"] = umax[4:5]
    ulb["mu"] = umin[5:6]
    uub["mu"] = umax[5:6]

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_ = np.vstack([X_star[iice[:,0],:]])
    u_ = np.vstack([u_star[iice[:,0],:]])

    # Getting the corresponding X_train and u_train(which is now scarce boundary/initial coordinates)
    X_train = {}
    u_train = {}

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    # velocity data
    if N_u:
        idx = np.random.choice(X_.shape[0], N_u, replace=False)
        X_train["uv"] = X_[idx,:]
        u_train["uv"] = u_[idx, 0:2]
    else:
        X_train["uv"] = X_bc
        u_train["uv"] = u_bc[:, 0:2]

    # surface elevation, always available, use the maximum points among all the other data set
    if N_s is None:
        Nlist = [N_u, N_H, N_C]
        N_s = max([i for i in Nlist if i is not None])

    idx = np.random.choice(X_.shape[0], N_s, replace=False)
    X_train["s"] = X_[idx,:]
    u_train["s"] = u_[idx, 2:3]

    # ice thickness, or bed elevation
    if N_H:
        idx = np.random.choice(X_.shape[0], N_H, replace=False)
        X_train["H"] = X_[idx,:]
        u_train["H"] = u_[idx, 3:4]
    else:
        if 'x_fl' in data.keys():
            # load thickness along flowlines
            H_fl = np.real(data['H_fl'].flatten()[:,None])
            x_fl = np.real(data['x_fl'].flatten()[:,None])
            y_fl = np.real(data['y_fl'].flatten()[:,None])
            X_fl = np.hstack((x_fl.flatten()[:,None], y_fl.flatten()[:,None]))

            X_train["H"] = np.vstack([X_bc, X_fl])
            u_train["H"] = np.vstack([u_bc[:, 3:4], H_fl])
        else:
            X_train["H"] = X_bc
            u_train["H"] = u_bc[:, 3:4]

    # friction coefficients
    if N_C:
        idx = np.random.choice(X_.shape[0], N_C, replace=False)
        X_train["C"] = X_[idx,:]
        u_train["C"] = u_[idx, 4:5]
    else:
        X_train["C"] = X_bc
        u_train["C"] = u_bc[:, 4:5]

    # mu
    if N_mu:
        idx = np.random.choice(X_.shape[0], N_mu, replace=False)
        X_train["mu"] = X_[idx,:]
        u_train["mu"] = u_[idx, 5:6]
    else:
        X_train["mu"] = X_bc
        u_train["mu"] = u_bc[:, 5:6]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    return x, y, Exact_vx, Exact_vy, X_star, u_star, X_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, None  #}}}
def prep_1D_data_all(path, N_f=None, N_u=None, N_s=None, N_H=None, N_C=None, FlightTrack=False): #{{{
    # Reading 1D SSA ref solutions: x-coordinate, provide ALL the variables in u_train
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

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = x

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vel.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] )) 

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0) 
    umin = u_star.min(axis=0)
    umax = u_star.max(axis=0) 
    ulb = {}
    uub = {}
    ulb["uv"] = umin[0:1]
    ulb["sH"] = umin[1:3]
    uub["sH"] = umax[1:3]
    ulb["C"] = umin[3:4]
    uub["C"] = umax[3:4]

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_ = np.vstack([X_star[iice[:,0],:]])
    u_ = np.vstack([u_star[iice[:,0],:]])

    # Getting the corresponding X_train and u_train(which is now scarce boundary/initial coordinates)
    X_train = {}
    u_train = {}

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    # velocity data
    if N_u:
        idx = np.random.choice(X_.shape[0], N_u, replace=False)
        X_train["uv"] = X_[idx,:]
        u_train["uv"] = u_[idx, 0:1]
    else:
        X_train["uv"] = X_bc
        u_train["uv"] = u_bc[:, 0:1]

    # surface elevation, always available, use the maximum points among all the other data set
    if N_s is None:
        Nlist = [N_u, N_H, N_C]
        N_s = max([i for i in Nlist if i is not None])

    idx = np.random.choice(X_.shape[0], N_s, replace=False)
    X_train["s"] = X_[idx,:]
    u_train["s"] = u_[idx, 1:2]

    # ice thickness, or bed elevation
    if N_H:
        idx = np.random.choice(X_.shape[0], N_H, replace=False)
        X_train["H"] = X_[idx,:]
        u_train["H"] = u_[idx, 2:3]
    else:
        X_train["H"] = X_bc
        u_train["H"] = u_bc[:, 2:3]

    # friction coefficients
    if N_C:
        idx = np.random.choice(X_.shape[0], N_C, replace=False)
        X_train["C"] = X_[idx,:]
        u_train["C"] = u_[idx, 3:4]
    else:
        X_train["C"] = X_bc
        u_train["C"] = u_bc[:, 3:4]

    # calving front info
    cx = data['cx'].flatten()[:,None]
    nx = data['nx'].flatten()[:,None]

    X_cf = cx
    n_cf = nx

    return x, Exact_vel, X_star, u_star, X_train, u_train, X_f, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
