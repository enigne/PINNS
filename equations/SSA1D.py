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
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred[:,3:4]) - tf.math.abs(u_star[:,3:4])) / tf.math.reduce_euclidean_norm(u_star[:,3:4])

    #}}}
class SSA1D_3NN_calvingfront_invertC(SSA1D): #{{{
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
                xub, xlb, uub[0:1], ulb[0:1],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            u = self.model(X)

            hsol = self.h_model(X)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

            C = self.C_model(X)

        u_x = tape.gradient(u, X)
        del tape

        return u, u_x, h, H, C

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 3:4]
        C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]

        u0_pred = self.model(X_u)
        hH_pred = self.h_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

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
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.C_model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,3:4])) / tf.math.reduce_euclidean_norm(u_star[:,3:4])

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        u_pred = self.model(X_star)

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary()

    #}}}
class SSA1D_3NN_solve_vel(SSA1D): #{{{
    '''
    class of inverting C from observed u, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-18]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:1], ulb[0:1],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            u = self.model(X)

            hsol = self.h_model(X)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

            C = self.C_model(X)

        u_x = tape.gradient(u, X)
        del tape

        return u, u_x, h, H, C

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |h-hobs|+|H-Hobs|+|C-Cobs|+|f1|
        '''
        # Dirichlet B.C. for u
        u0 = self.u_bc[:, 0:1]
        u0_pred = self.model(self.X_bc)

        # match h, H, and C to the training data
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]
        C0 = uv[:,3:4]

        hH_pred = self.h_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]
        C0_pred = self.C_model(X_u)

        # f_model on the collocation points 
        f1_pred = self.f_model()
        # calving front
        fc1_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit at B.C.
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
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of u
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,0:1])) / tf.math.reduce_euclidean_norm(u_star[:,0:1])

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        u_pred = self.model(X_star)

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary()

    #}}}
class SSA1D_frictionNN_saved(SSA1D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:1], ulb[0:1],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        # hp["friction_layers"] defines C model
        fri_lb = (ulb[3:4]**2)*(ulb[0:1]**(1.0/n))
        fri_ub = (uub[3:4]**2)*(uub[0:1]**(1.0/n))
        self.friction_model = create_NN(hp["friction_layers"], inputRange=(np.concatenate([ulb[0:1],ulb[3:4]]), np.concatenate([uub[0:1],uub[3:4]])), outputRange=(fri_lb, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.friction_model.trainable_variables

#         self.friction_model = create_NN(hp["friction_layers"], inputRange=(ulb[0:3], uub[0:3]), outputRange=(fri_lb, fri_ub))
#         self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
#         self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            u = self.model(X)

            hsol = self.h_model(X)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

            C = self.C_model(X)

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

            # use NN to predict the basal stress
            tempX = tf.concat([u, C], axis=1)
            taub = self.friction_model(tempX) 

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)

        # surface gradient
        h_x = tape.gradient(h, self.x_f)

        # Letting the tape go
        del tape

        u_norm = (u**2)**0.5
        f1 = sigma11 - taub*u/(u_norm+1e-30) - self.rhoi*self.g*H*h_x
        return f1


    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
#         C0 = self.u_bc[:, 3:4]
#         C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]
        C0 = uv[:,3:4]

        u0_pred = self.model(X_u)
        hH_pred = self.h_model(X_u)
        C0_pred = self.C_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

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
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of C, since C^2 in the friction law, the sign of C does not matter
        '''
        sol_pred = self.C_model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,3:4])) / tf.math.reduce_euclidean_norm(u_star[:,3:4])

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        u_pred = self.model(X_star)

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()

    #}}}
class SSA1D_frictionNN(SSA1D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f,
            X_bc, u_bc, X_cf, n_cf,
            xub, xlb, uub, ulb,
            modelPath, reloadModel,
            mu, n=3.0,
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, X_f,
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:1], ulb[0:1],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        # hp["friction_layers"] defines C model
        fri_lb = (ulb[3:4]**2)*(ulb[0:1]**(1.0/n))
        fri_ub = (uub[3:4]**2)*(uub[0:1]**(1.0/n))
        self.friction_model = create_NN(hp["friction_layers"], inputRange=(np.concatenate([ulb[0:1],ulb[3:4]]), np.concatenate([uub[0:1],uub[3:4]])), outputRange=(fri_lb, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.friction_model.trainable_variables

#         self.friction_model = create_NN(hp["friction_layers"], inputRange=(ulb[0:3], uub[0:3]), outputRange=(fri_lb, fri_ub))
#         self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
#         self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            u = self.model(X)

            hsol = self.h_model(X)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

            C = self.C_model(X)

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

            # use NN to predict the basal stress
            tempX = tf.concat([u, C], axis=1)
            taub = self.friction_model(tempX)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)

        # surface gradient
        h_x = tape.gradient(h, self.x_f)

        # Letting the tape go
        del tape

        u_norm = (u**2)**0.5
        f1 = sigma11 - taub*u/(u_norm+1e-30) - self.rhoi*self.g*H*h_x
        return f1


    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
#         C0 = self.u_bc[:, 3:4]
#         C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]
        C0 = uv[:,3:4]

        u0_pred = self.model(X_u)
        hH_pred = self.h_model(X_u)
        C0_pred = self.C_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

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
        totalloss = mse_u + mse_h + mse_H + mse_C + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h,
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_fc1": mse_fc1}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        u_pred = self.model(X_star)
        C_pred = self.C_model(X_star)
        uC_pred = tf.concat([u_pred, C_pred], axis=1)
        sol_pred = self.friction_model(uC_pred)

        # ref taub
        ref_sol = u_star[:,3:4]**2**(u_star[:,0:1]**(1.0/3.0))

        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        u_pred = self.model(X_star)

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()

    #}}}
class SSA1D_frictionNN_uhH(SSA1D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-2, 1e-6, 1e-10, 1e-10, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:1], ulb[0:1],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[1:3], uub[1:3]))
        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[3:4], uub[3:4]))

        # hp["friction_layers"] defines friction model
        fri_lb = (ulb[3:4]**2)*(ulb[0:1]**(1.0/n))
        fri_ub = (uub[3:4]**2)*(uub[0:1]**(1.0/n))

        self.friction_model = create_NN(hp["friction_layers"], inputRange=(ulb[0:3], uub[0:3]), outputRange=(fri_lb, fri_ub))
        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)

            u = self.model(X)

            hsol = self.h_model(X)
            h = hsol[:, 0:1]
            H = hsol[:, 1:2]

        u_x = tape.gradient(u, X)
        del tape

        return u, u_x, h, H, 0

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

            # use NN to predict the basal stress
            tempX = tf.concat([u, h, H], axis=1)
            taub = self.friction_model(tempX) 

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)

        # surface gradient
        h_x = tape.gradient(h, self.x_f)

        # Letting the tape go
        del tape

        u_norm = (u**2)**0.5
        f1 = sigma11 - taub*u/(u_norm+1e-30) - self.rhoi*self.g*H*h_x
        return f1


    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        h0 = uv[:,1:2]
        H0 = uv[:,2:3]

        u0_pred = self.model(X_u)
        hH_pred = self.h_model(X_u)
        h0_pred = hH_pred[:,0:1]
        H0_pred = hH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred = self.f_model()
        # calving front
        fc1_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        # geometry misfit
        mse_h = self.loss_weights[1]*tf.reduce_mean(tf.square(h0 - h0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))

        # sum the total
        totalloss = mse_u + mse_h + mse_H + mse_f1 + mse_fc1
        return {"loss": totalloss, "mse_u": mse_u, "mse_h": mse_h, 
                "mse_H": mse_H, "mse_f1": mse_f1, "mse_fc1": mse_fc1} 

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        u_pred = self.model(X_star)
        hH_pred = self.h_model(X_star)
        uhH_pred = tf.concat([u_pred, hH_pred], axis=1)
        sol_pred = self.friction_model(uhH_pred)

        # ref taub
        ref_sol = u_star[:,3:4]**2**(u_star[:,0:1]**(1.0/3.0))

        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        u_pred = self.model(X_star)

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()

    #}}}

