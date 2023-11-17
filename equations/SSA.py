import sys
import os
import tensorflow as tf
import numpy as np

from utils import *

class SSA_3NN(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 2D problem:
        use one NN with 2 inputs: [x,y], 5 outputs [u, v, s, H, C] to learn SSA
        The loss_weights are for: [u,v], [s,H], [C], [f1, f2], [fc1, fc2]
    '''
    def __init__(self, hp, logger, X_f, 
            X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-10, 1e-16]):
        super().__init__(hp, logger, xub, xlb, uub["uv"], ulb["uv"], modelPath+"/model/", reloadModel=reloadModel)

        # weights of the loss functions
        self.loss_weights = tf.constant(loss_weights, dtype=self.dtype)
            
        # scaling factors
        self.ub = tf.constant(xub, dtype=self.dtype)
        self.lb = tf.constant(xlb, dtype=self.dtype)

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

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb["sH"], uub["sH"]))
            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb["C"], uub["C"]))

        # set trainable variables
        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    def fit(self, X_u, u):
        '''
        main function to run the trainning: Adams + L-BFGS
        '''
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = {x:self.tensor(X_u[x]) for x in X_u.keys()}
        u = {x:self.tensor(u[x]) for x in u.keys()}

        # Optimizing
        self.Adam_optimization(X_u, u)

        # use LBFGS
        self.LBFGS_optimization(X_u, u)

        # log
        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            uv_sol = self.model(Xtemp)
            u = uv_sol[:, 0:1]
            v = uv_sol[:, 1:2]

            sH_sol = self.h_model(Xtemp)
            s = sH_sol[:, 0:1]
            H = sH_sol[:, 1:2]

            C = self.C_model(Xtemp)

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, C

    @tf.function
    def f_model(self):
        '''
        The physical laws in PINN, SSA 2D
        '''
        # viscosity
        mu = self.mu
        n = self.n

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and y
            tape.watch(self.x_f)
            tape.watch(self.y_f)

            # just rename the input
            X_f = tf.concat([self.x_f, self.y_f], axis=1)

            # Getting the prediction
            u, v, u_x, v_x, u_y, v_y, s, H, C = self.nn_model(X_f)

            eta = 0.5*mu *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = eta * H
            B11 = etaH*(4*u_x + 2*v_y)
            B22 = etaH*(4*v_y + 2*u_x)
            B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)
        sigma12 = tape.gradient(B12, self.y_f)

        sigma21 = tape.gradient(B12, self.x_f)
        sigma22 = tape.gradient(B22, self.y_f)
                
        # surface gradient
        s_x = tape.gradient(s, self.x_f)
        s_y = tape.gradient(s, self.y_f)

        # Letting the tape go
        del tape

        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2

    # Calving front condition
    @tf.function
    def cf_model (self, X, nn):
        '''
        function for calving front boundary
        '''
        # normal direction
        nx = nn[:, 0:1]
        ny = nn[:, 1:2]

        # viscosity
        mu = self.mu
        n = self.n

        # velocity component
        u, v, u_x, v_x, u_y, v_y, s, H, C = self.nn_model(X)
        base = s - H

        # viscosity
        eta = 0.5*mu *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # Calving front condition
        fc1 = B11*nx + B12*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*nx
        fc2 = B12*nx + B22*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*ny

        return fc1, fc2

    @tf.function
    def loss(self, u_train, X_train):
        '''
        The loss function: fit the data given in u on X_u, and minimize the residual of the PDE on predefined X_f
        '''
        # match s, H, and C to the training data
        uv0 = u_train["uv"]
        u0 = uv0[:,0:1]
        v0 = uv0[:,1:2]
        X_uv = X_train["uv"]
        uv_pred = self.model(X_uv)
        u0_pred = uv_pred[:, 0:1]
        v0_pred = uv_pred[:, 1:2]

        # s and H can be from different data sources
        s0 = u_train["s"]
        X_s = X_train["s"]
        s_pred = self.h_model(X_s)
        s0_pred = s_pred[:, 0:1]

        H0 = u_train["H"]
        X_H = X_train["H"]
        H_pred = self.h_model(X_H)
        H0_pred = H_pred[:, 1:2]

        C0 = u_train["C"]
        X_C = X_train["C"]
        C0_pred = self.C_model(X_C)

        # f_model on the collocation points 
        f1_pred, f2_pred= self.f_model()

        # Calving on X_cf
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # friction misfit
        mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_f1 + mse_f2 + mse_s + mse_H + mse_C + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2, 
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:, 0:1]
        v_pred = uv_pred[:, 1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        C_pred = self.C_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    @tf.function
    def H_test_error(self, X_test, u_test):
        '''
        test error of H, second output from h_model
        '''
        sol_pred = self.h_model(X_test)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred[:,1:2]) - tf.math.abs(u_test)) / tf.math.reduce_euclidean_norm(u_test)

    @tf.function
    def C_test_error(self, X_test, u_test):
        '''
        test error of C
        '''
        sol_pred = self.C_model(X_test)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_test)) / tf.math.reduce_euclidean_norm(u_test)

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary()

    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA_4NN(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 2D problem:
        use 4-NNs with 2 inputs: [x,y], with outputs [u, v], [s, H], [C], [mu] to learn SSA
        The loss_weights are for: "uv":[u,v], "sH":[s,H], "C":[C], "mu":[mu], "f":[f1, f2], "fc":[fc1, fc2]
    '''
    def __init__(self, hp, logger, X_f, 
            X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu=None, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-10, 1e-16]):
        super().__init__(hp, logger, xub, xlb, uub["uv"], ulb["uv"], modelPath+"/model/", reloadModel=reloadModel)

        # weights of the loss functions
        self.loss_weights = tf.constant(loss_weights, dtype=self.dtype)
            
        # scaling factors
        self.ub = tf.constant(xub, dtype=self.dtype)
        self.lb = tf.constant(xlb, dtype=self.dtype)

        # Calving front
        self.X_cf = self.tensor(X_cf)
        self.n_cf = self.tensor(n_cf)

        # viscosity eponent
        self.n = tf.constant(n, dtype=self.dtype)

        # some constants
        self.rhoi = tf.constant(917, dtype=self.dtype)  # kg/m^3
        self.rhow = tf.constant(1023, dtype=self.dtype) # kg/m^3
        self.g = tf.constant(9.81, dtype=self.dtype)    # m/s^2
        self.yts = tf.constant(3600.0*24*365, dtype=self.dtype)

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
            self.mu_model = tf.keras.models.load_model(modelPath+"/mu_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb["sH"], uub["sH"]))
            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb["C"], uub["C"]))
            # hp["mu_layers"] defines mu model
            self.mu_model = create_NN(hp["mu_layers"], inputRange=(xlb, xub), outputRange=(ulb["mu"], uub["mu"]))

        # set trainable variables
        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.mu_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.mu_model.trainable_variables

    def fit(self, X_u, u):
        '''
        main function to run the trainning: Adams + L-BFGS
        '''
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = {x:self.tensor(X_u[x]) for x in X_u.keys()}
        u = {x:self.tensor(u[x]) for x in u.keys()}

        # Optimizing
        self.Adam_optimization(X_u, u)

        # use LBFGS
        self.LBFGS_optimization(X_u, u)

        # log
        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            uv_sol = self.model(Xtemp)
            u = uv_sol[:, 0:1]
            v = uv_sol[:, 1:2]

            sH_sol = self.h_model(Xtemp)
            s = sH_sol[:, 0:1]
            H = sH_sol[:, 1:2]

            C = self.C_model(Xtemp)

            mu = self.mu_model(Xtemp)

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, C, mu

    @tf.function
    def f_model(self):
        '''
        The physical laws in PINN, SSA 2D
        '''
        # viscosity exponent only
        n = self.n

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and y
            tape.watch(self.x_f)
            tape.watch(self.y_f)

            # just rename the input
            X_f = tf.concat([self.x_f, self.y_f], axis=1)

            # Getting the prediction
            u, v, u_x, v_x, u_y, v_y, s, H, C, mu = self.nn_model(X_f)

            eta = 0.5*mu *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = eta * H
            B11 = etaH*(4*u_x + 2*v_y)
            B22 = etaH*(4*v_y + 2*u_x)
            B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)
        sigma12 = tape.gradient(B12, self.y_f)

        sigma21 = tape.gradient(B12, self.x_f)
        sigma22 = tape.gradient(B22, self.y_f)
                
        # surface gradient
        s_x = tape.gradient(s, self.x_f)
        s_y = tape.gradient(s, self.y_f)

        # Letting the tape go
        del tape

        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2

    # Calving front condition
    @tf.function
    def cf_model (self, X, nn):
        '''
        function for calving front boundary
        '''
        # normal direction
        nx = nn[:, 0:1]
        ny = nn[:, 1:2]

        # viscosity exponent
        n = self.n

        # velocity component
        u, v, u_x, v_x, u_y, v_y, s, H, C, mu = self.nn_model(X)
        base = s - H

        # viscosity
        eta = 0.5*mu *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # Calving front condition
        fc1 = B11*nx + B12*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*nx
        fc2 = B12*nx + B22*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*ny

        return fc1, fc2

    @tf.function
    def loss(self, u_train, X_train):
        '''
        The loss function: fit the data given in u on X_u, and minimize the residual of the PDE on predefined X_f
        '''
        # match s, H, and C to the training data
        uv0 = u_train["uv"]
        u0 = uv0[:,0:1]
        v0 = uv0[:,1:2]
        X_uv = X_train["uv"]
        uv_pred = self.model(X_uv)
        u0_pred = uv_pred[:, 0:1]
        v0_pred = uv_pred[:, 1:2]

        # s and H can be from different data sources
        s0 = u_train["s"]
        X_s = X_train["s"]
        s_pred = self.h_model(X_s)
        s0_pred = s_pred[:, 0:1]

        H0 = u_train["H"]
        X_H = X_train["H"]
        H_pred = self.h_model(X_H)
        H0_pred = H_pred[:, 1:2]

        # friction coefficient
        C0 = u_train["C"]
        X_C = X_train["C"]
        C0_pred = self.C_model(X_C)

        # rheologyB
        mu0 = u_train["mu"]
        X_mu = X_train["mu"]
        mu0_pred = self.mu_model(X_mu)

        # f_model on the collocation points 
        f1_pred, f2_pred= self.f_model()

        # Calving on X_cf
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # friction misfit
        mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))
        # mu misfit
        mse_mu = self.loss_weights[5]*tf.reduce_mean(tf.square(mu0 - mu0_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_f1 + mse_f2 + mse_s + mse_H + mse_C + mse_fc1 + mse_fc2 + mse_mu
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_mu":mse_mu, "mse_f1": mse_f1, "mse_f2": mse_f2, 
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:, 0:1]
        v_pred = uv_pred[:, 1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        C_pred = self.C_model(X_star)

        mu_pred = self.mu_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), mu_pred.numpy()

    @tf.function
    def H_test_error(self, X_test, u_test):
        '''
        test error of H, second output from h_model
        '''
        sol_pred = self.h_model(X_test)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred[:,1:2]) - tf.math.abs(u_test)) / tf.math.reduce_euclidean_norm(u_test)

    @tf.function
    def C_test_error(self, X_test, u_test):
        '''
        test error of C
        '''
        sol_pred = self.C_model(X_test)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_test)) / tf.math.reduce_euclidean_norm(u_test)

    @tf.function
    def mu_test_error(self, X_test, u_test):
        '''
        test error of mu
        '''
        sol_pred = self.mu_model(X_test)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_test)) / tf.math.reduce_euclidean_norm(u_test)

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.mu_model.summary()

    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.mu_model.save(self.modelPath+"/mu_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
