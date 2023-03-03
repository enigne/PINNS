import sys
import os
import tensorflow as tf
import numpy as np

from utils import *

class SSA1D(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 1D problem:
        use one NN with 1 inputs: [x], 4 outputs [u, h, H, C] to learn SSA
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, xub, xlb, uub, ulb, modelPath, reloadModel=reloadModel)

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

        # save the collocation points
        self.x_f = self.tensor(X_f)

    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            sol = self.model(X)
            u = sol[:, 0:1]
            h = sol[:, 1:2]
            H = sol[:, 2:3]
            C = sol[:, 3:4]

        u_x = tape.gradient(u, X)
        del tape

        return u, u_x, h, H, C

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

            # just rename the input
            X_f = self.x_f

            # Getting the prediction
            u, u_x, h, H, C = self.nn_model(X_f)

            eta = 0.5*mu *(u_x**2+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = eta * H
            B11 = etaH*(4*u_x)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)
                
        # surface gradient
        h_x = tape.gradient(h, self.x_f)

        # Letting the tape go
        del tape

        # compute the basal stress
        u_norm = (u**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*h_x

        return f1

    # Calving front condition
    @tf.function
    def cf_model (self, X, nn):
        '''
        function for calving front boundary
        '''
        # nn is canceled in 1D problem
        # nx = nn[:, 0:1]

        # viscosity
        mu = self.mu
        n = self.n

        # velocity component
        u, u_x, h, H, C = self.nn_model(X)
        base = h - H

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
        The basic format of loss function: knowing the geometry and C, solve for u and v
        '''
        # Dirichlet B.C. for u and v
        u0 = self.u_bc[:, 0:1]
        sol_bc_pred = self.model(self.X_bc)
        u0_pred = sol_bc_pred[:,0:1]
        
        # match H, bed, and C to the training data
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]
        C0 = uv[:,3:4]
        uv_pred = self.model(X_u)
        h0_pred = uv_pred[:,1:2]
        H0_pred = uv_pred[:,2:3]
        C0_pred = uv_pred[:,3:4]

        # f_model on the collocation points 
        f1_pred = self.f_model()

        # Calving on X_cf
        fc1_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        # geometry misfit
        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # friction misfit
        mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))

        # sum the total
        totalloss = mse_u + mse_f1 + mse_h + mse_H + mse_C
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        sol_pred = self.model(X_star)
        u_pred = sol_pred[:, 0:1]
        h_pred = sol_pred[:, 1:2]
        H_pred = sol_pred[:, 2:3]
        C_pred = sol_pred[:, 3:4]
        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        return the test error of u 
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(sol_pred[:,0:1] - u_star[:,0:1]) / tf.math.reduce_euclidean_norm(u_star[:,0:1])
    #}}}
class SSA1D_invertC(SSA1D): #{{{
    '''
    class of inverting C from observed u, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub, ulb,
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)

    # only need to overwrite the loss function, change it from inferring u --> inferring C
    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 3:4]
        sol_bc_pred = self.model(self.X_bc)
        C0_pred = sol_bc_pred[:,3:4]

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        h0_pred = uv_pred[:,1:2]
        H0_pred = uv_pred[:,2:3]

        # f_model on the collocation points 
        f1_pred = self.f_model()

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        # geometry misfit
        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # friction misfit
        mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        # calving front boundary

        # sum the total
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred[:,3:4]) - tf.math.abs(u_star[:,3:4])) / tf.math.reduce_euclidean_norm(u_star[:,3:4])

    #}}}
class SSA1D_calvingfront_invertC(SSA1D): #{{{
    '''
    class of inverting C from observed u, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub, ulb,
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)

    # only need to overwrite the loss function, change it from inferring u --> inferring C
    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 3:4]
        sol_bc_pred = self.model(self.X_bc)
        C0_pred = sol_bc_pred[:,3:4]

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        h0_pred = uv_pred[:,1:2]
        H0_pred = uv_pred[:,2:3]

        # f_model on the collocation points 
        f1_pred = self.f_model()
        # calving front
        fc1_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        # geometry misfit
        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # friction misfit
        mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))

        # sum the total
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1"} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred[:,3:4]) - tf.math.abs(u_star[:,3:4])) / tf.math.reduce_euclidean_norm(u_star[:,3:4])

    #}}}
class SSA3NN_invertC(SSA1D): #{{{
    '''
    class of inverting C from observed u and v, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["layers"] defines uv model

        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) 
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # get the velocity and derivative information
    @tf.function
    def nn_model(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            uvsol = self.model(Xtemp)
            u = uvsol[:, 0:1]
            v = uvsol[:, 1:2]

            hsol = self.h_model(Xtemp)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

            C = self.C_model(Xtemp)

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, h, H, C


    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 4:5]
        C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        h0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]
        hH_pred = self.h_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()

        # Calving on X_cf
       # fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # misfits
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))

        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        mse_C = self.loss_weights[2]* tf.reduce_mean(tf.square(C0 - C0_pred))

        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_f1 + mse_f2 + mse_h + mse_H + mse_C
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.C_model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,4:5])) / tf.math.reduce_euclidean_norm(u_star[:,4:5])

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        sol_pred = self.model(X_star)
        u_pred = sol_pred[:, 0:1]
        v_pred = sol_pred[:, 1:2]

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary()
    #}}}
class SSA3NN_calvingfront_invertC(SSA3NN_invertC): #{{{
    '''
    class of inverting C from observed u and v, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-1, 1e-2, 1e-2, 1e-4, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub, ulb,
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 4:5]
        C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        h0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]
        hH_pred = self.h_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()

        # Calving on X_cf
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # misfits
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))

        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        mse_C = self.loss_weights[2]* tf.reduce_mean(tf.square(C0 - C0_pred))

        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))

        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))
        # sum the total
        totalloss = mse_u + mse_v + mse_f1 + mse_f2 + mse_h + mse_H + mse_C + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mes_fc1": mse_fc1, "mse_fc2": mse_fc2} 
    #}}}
