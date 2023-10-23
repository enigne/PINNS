import sys
import os
import tensorflow as tf
import numpy as np

from utils import *

class SSA1D_transient(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 1D time depented problem:
        use one NN with 2 inputs: [x,t], 5 outputs [u, s, H, C, smb] to learn SSA
        The loss_weights are for: [u], [s,H], [C], [fSSA], [smb, fH], [fc1, fc2]
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-5, 1e-14]):
        super().__init__(hp, logger, xub, xlb, uub[0:1], ulb[0:1], modelPath, reloadModel=reloadModel) 
        # This constructor will only create a NN according to 'layer', which will be the [u]-net

        # weights of the loss functions
        self.loss_weights = tf.constant(loss_weights, dtype=self.dtype)
            
        # scaling factors
        self.ub = tf.constant(xub, dtype=self.dtype)
        self.lb = tf.constant(xlb, dtype=self.dtype)

        # Dirichlet B.C.
        self.X_bc = self.tensor(X_bc)
        self.u_bc = self.tensor(u_bc)

        # Calving front
        self.X_cf = self.tensor(X_cf)
        self.n_cf = self.tensor(n_cf)

        # viscosity
        self.mu = tf.constant(mu, dtype=self.dtype)
        self.n = tf.constant(n, dtype=self.dtype)

        # some constants
        self.rhoi = tf.constant(917, dtype=self.dtype)  # kg/m^3
        self.rhow = tf.constant(1023, dtype=self.dtype) # kg/m^3
        self.g = tf.constant(9.81, dtype=self.dtype)    # m/s^2
        self.yts = tf.constant(3600.0*24*365, dtype=self.dtype)

        # Separating the collocation coordinates: x and t
        self.x_f = self.tensor(X_f[:, 0:1])
        self.t_f = self.tensor(X_f[:, 1:2])

        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb[0:1], xub[0:1]), outputRange=(ulb[4:5], uub[4:5]))

        # hp["smb_layers"] defines smb model
        self.smb_model = create_NN(hp["smb_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.smb_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.smb_model.trainable_variables

    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        x = X[:, 0:1]
        t = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            Xtemp = tf.concat([x, t], axis=1)

            uv_sol = self.model(Xtemp)
            u = uv_sol[:, 0:1]

            sH_sol = self.h_model(Xtemp)
            s = sH_sol[:, 0:1]
            H = sH_sol[:, 1:2]

            smb = self.smb_model(Xtemp)

            C = self.C_model(x)

        u_x = tape.gradient(u, x)
        H_x = tape.gradient(H, x)
        H_t = tape.gradient(H, t)
        del tape

        return u, s, H, C, smb, u_x, H_x, H_t

    @tf.function
    def f_model(self):
        '''
        The actual PINN
        '''
        # viscosity
        mu = self.mu
        n = self.n

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and y
            tape.watch(self.x_f)
            tape.watch(self.t_f)

            # just rename the input
            X_f = tf.concat([self.x_f, self.t_f], axis=1)

            # Getting the prediction
            u, s, H, C, smb, u_x, H_x, H_t = self.nn_model(X_f)

            eta = 0.5*mu *(u_x**2+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = eta * H
            B11 = etaH*(4*u_x)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)

        # surface gradient
        s_x = tape.gradient(s, self.x_f)

        # Letting the tape go
        del tape

        # compute the basal stress
        u_norm = (u**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        fSSA = sigma11 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        fH = H_t + u_x*H + H_x*u - smb

        return fSSA, fH

    # Calving front condition
    @tf.function
    def cf_model (self, X, nn):
        '''
        function for calving front boundary
        '''
        # viscosity
        mu = self.mu
        n = self.n

        # velocity component
        u, s, H, C, smb, u_x, H_x, H_t = self.nn_model(X)
        base = s - H

        # viscosity
        eta = 0.5*mu *(u_x**2+1.0e-30)**(0.5*(1.0-n)/n)

        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x)

        # Calving front condition
        fc1 = B11 - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)

        return fc1

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss function
        '''
        # match u, v, s, H, and C to the training data
        u0 = uv[:,0:1]
        s0 = uv[:,1:2]
        H0 = uv[:,2:3]
        smb0 = uv[:,3:4]

#         # Dirichlet B.C. for C
#         C0 = self.u_bc[:, 3:4]
#         C0_pred = self.C_model(self.X_bc)

        # match with data
        u_pred = self.model(X_u)

        sH_pred = self.h_model(X_u)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        smb_pred = self.smb_model(X_u)

        # f_model on the collocation points 
        fSSA_pred, fH_pred= self.f_model()

        # Calving on X_cf
        fc1_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H_pred))
        # friction misfit
#         mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_fSSA = self.loss_weights[3]*tf.reduce_mean(tf.square(fSSA_pred))
        # thickness equation and smb
        mse_smb = self.loss_weights[4]*(self.yts**2)*tf.reduce_mean(tf.square(smb0 - smb_pred))
        mse_fH = self.loss_weights[4]*(self.yts**2)*tf.reduce_mean(tf.square(fH_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[5]*tf.reduce_mean(tf.square(fc1_pred))

        # sum the total
        totalloss = mse_u + mse_fSSA + mse_fH + mse_s + mse_H + mse_smb + mse_fc1 #+ mse_C
        return {"loss": totalloss, "mse_u": mse_u, "mse_s": mse_s, "mse_H": mse_H,  #"mse_C": mse_C,
                "mse_smb": mse_smb, "mse_fSSA": mse_fSSA, "mse_fH": mse_fH, "mse_fc1": mse_fc1} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:, 0:1]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        C_pred = self.C_model(X_star[:,0:1])

        smb_pred = self.smb_model(X_star)

        return u_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), smb_pred.numpy()

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        return the test error of C 
        '''
        sol_pred = self.C_model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star)) / tf.math.reduce_euclidean_norm(u_star)

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.smb_model.summary()
    #}}}
