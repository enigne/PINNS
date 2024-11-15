import sys
import os
import tensorflow as tf
import numpy as np

from utils import *

class SSA2D(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 2D problem:
        use one NN with 2 inputs: [x,y], 5 outputs [u, v, s, H, C] to learn SSA
        The loss_weights are for: [u,v], [s,H], [C], [f1, f2], [fc1, fc2]
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-8]):
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

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

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

            sol = self.model(Xtemp)
            u = sol[:, 0:1]
            v = sol[:, 1:2]
            s = sol[:, 2:3]
            H = sol[:, 3:4]
            C = sol[:, 4:5]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, C

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
    def loss(self, uv, X_u):
        '''
        The basic format of loss function: knowing the geometry and C, solve for u and v
        '''
        # Dirichlet B.C. for u and v
        u0 = self.u_bc[:, 0:1]
        v0 = self.u_bc[:, 1:2]
        sol_bc_pred = self.model(self.X_bc)
        u0_pred = sol_bc_pred[:,0:1]
        v0_pred = sol_bc_pred[:,1:2]
        
        # match s, H, and C to the training data
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]
        C0 = uv[:,4:5]
        uv_pred = self.model(X_u)
        s0_pred = uv_pred[:,2:3]
        H0_pred = uv_pred[:,3:4]
        C0_pred = uv_pred[:,4:5]

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
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2, "mse_fc1": mse_fc1, "mse_fc2": mse_fc2} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        sol_pred = self.model(X_star)
        u_pred = sol_pred[:, 0:1]
        v_pred = sol_pred[:, 1:2]
        s_pred = sol_pred[:, 2:3]
        H_pred = sol_pred[:, 3:4]
        C_pred = sol_pred[:, 4:5]
        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        return the test error of u 
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(sol_pred[:,0:2] - u_star[:,0:2]) / tf.math.reduce_euclidean_norm(u_star[:,0:2])
    #}}}
class SSA2D_3NN_calvingfront_invertC(SSA2D): #{{{
    '''
    class of inverting C from observed u, as well as h and H, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|h-hobs|+|H-Hobs|+|f1|
        '''
        # Dirichlet B.C. for C
        C0 = self.u_bc[:, 4:5]
        C0_pred = self.C_model(self.X_bc)

        # match u, v, s, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
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
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_C + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s, 
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2, 
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2} 

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
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:, 0:1]
        v_pred = uv_pred[:, 1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy()

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
class SSA2D_3NN_solve_vel(SSA2D): #{{{
    '''
    class of solving u,v from s, H, C, with no calving front boundary
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-18]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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
    def loss(self, uv, X_u):
        '''
        loss = |h-hobs|+|H-Hobs|+|C-Cobs|+|f1|
        '''
        # Dirichlet B.C. for u
        u0 = self.u_bc[:, 0:1]
        v0 = self.u_bc[:, 1:2]
        uv_pred = self.model(self.X_bc)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        # match h, H, and C to the training data
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]
        C0 = uv[:,4:5]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]
        C0_pred = self.C_model(X_u)

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
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
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_C + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of u
        '''
        sol_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,0:2])) / tf.math.reduce_euclidean_norm(u_star[:,0:2])

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
class SSA2D_frictionNN(SSA2D): #{{{
    '''
    class of learning friction laws from observed u, v and C, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath, reloadModel,
                mu, loss_weights=loss_weights)
        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        # hp["friction_layers"] defines friction model
        fri_lb = (ulb[4:5]**2)*((ulb[0:1]**2.0+ulb[1:2]**2.0)**(0.5/n))
        fri_ub = (uub[4:5]**2)*((uub[0:1]**2.0+uub[1:2]**2.0)**(0.5/n))
        self.friction_model = create_NN(hp["friction_layers"], inputRange=(np.concatenate([ulb[0:2],ulb[4:5]]), np.concatenate([uub[0:2],uub[4:5]])), outputRange=(fri_lb, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.friction_model.trainable_variables
        
    # need to overwrite nn_model, which is used in computing the loss function
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
                        
            # use NN to predict the basal stress
            tempX = tf.concat([u, v, C], axis=1)
            taub = self.friction_model(tempX) 

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

        f1 = sigma11 + sigma12 - taub*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - taub*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2
    
    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]
        C0 = uv[:,4:5]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]

        C0_pred = self.C_model(X_u)

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
        fc1_pred,fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
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
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_C + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:,0:1]
        v_pred = uv_pred[:,1:2]
        C_pred = self.C_model(X_star)
        
        uC_pred = tf.concat([u_pred, v_pred, C_pred], axis=1)
        sol_pred = self.friction_model(uC_pred)
        
        # ref taub
        ref_sol = u_star[:,4:5]**2*((u_star[:,0:1]**2.0+u_star[:,1:2]**2.0)**(1.0/6.0))
        
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)
    
    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:, 0:1]
        v_pred = uv_pred[:, 1:2]

        hH_pred = self.h_model(X_star)
        h_pred = hH_pred[:, 0:1]
        H_pred = hH_pred[:, 1:2]
        C_pred = self.C_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), h_pred.numpy(), H_pred.numpy(), C_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()

    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.friction_model.save(self.modelPath+"/friction_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA2D_transient(NeuralNetwork): #{{{
    '''
    The main class of PINN-SSA, for 2D time dependent problem:
        use one NN with 3 inputs: [x,y,t], s65 outputs [u, v, s, H, C, smb] to learn SSA
        The loss_weights are for: [u,v], [s,H], [C], [fSSA1, fSSA2], [smb, fH], [fc1, fc2]
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-1, 1e-8]):
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

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])
        self.t_f = self.tensor(X_f[:, 2:3])

        # hp["h_layers"] defines h and H model
        self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

        # hp["smb_layers"] defines smb model
        self.smb_model = create_NN(hp["smb_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        # hp["C_layers"] defines C model
        self.C_model = create_NN(hp["C_layers"], inputRange=(xlb[0:2], xub[0:2]), outputRange=(ulb[5:6], uub[5:6]))


        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1]) + (self.smb_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables + self.smb_model.trainable_variables

    @tf.function
    def nn_model(self, X):
        '''
        get the velocity and derivative prediction from the NN
        '''
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            XYTtemp = tf.concat([x, y, t], axis=1)

            uv_sol = self.model(XYTtemp)
            u = uv_sol[:, 0:1]
            v = uv_sol[:, 1:2]

            sH_sol = self.h_model(XYTtemp)
            s = sH_sol[:, 0:1]
            H = sH_sol[:, 1:2]

            smb = self.smb_model(XYTtemp)

            XYtemp = tf.concat([x, y], axis=1)
            C = self.C_model(XYtemp)

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        H_x = tape.gradient(H, x)
        H_y = tape.gradient(H, y)
        H_t = tape.gradient(H, t)
        del tape

        return u, v, s, H, C, smb, u_x, v_x, u_y, v_y, H_x, H_y, H_t

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
            # Watching the inputs we’ll need later, x, y, t
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            tape.watch(self.t_f)

            # just rename the input
            X_f = tf.concat([self.x_f, self.y_f, self.t_f], axis=1)

            # Getting the prediction
            u, v, s, H, C, smb, u_x, v_x, u_y, v_y, H_x, H_y, H_t = self.nn_model(X_f)

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

        fSSA1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        fSSA2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        fH = H_t + u_x*H + u*H_x + v_y*H + v*H_y - smb

        return fSSA1, fSSA2, fH

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
        u, v, s, H, C, smb, u_x, v_x, u_y, v_y, H_x, H_y, H_t = self.nn_model(X)
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
    def loss(self, uv, X_u):
        '''
        loss function
        '''
        # match s, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]
        smb0 = uv[:,4:5]

        uv_pred = self.model(X_u)
        u_pred = uv_pred[:,0:1]
        v_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        smb_pred = self.smb_model(X_u)

        # f_model on the collocation points 
        fSSA1_pred, fSSA2_pred, fH_pred = self.f_model()

        # Calving on X_cf
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H_pred))
        # friction misfit
        #mse_C = self.loss_weights[2]*tf.reduce_mean(tf.square(C0 - C_pred))
        # residual of PDE
        mse_fSSA1 = self.loss_weights[3]*tf.reduce_mean(tf.square(fSSA1_pred))
        mse_fSSA2 = self.loss_weights[3]*tf.reduce_mean(tf.square(fSSA2_pred))
        # thickness equation and smb
        mse_smb = self.loss_weights[4]*(self.yts**2)*tf.reduce_mean(tf.square(smb0 - smb_pred))
        mse_fH = self.loss_weights[4]*(self.yts**2)*tf.reduce_mean(tf.square(fH_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_fSSA1 + mse_fSSA2 + mse_fH + mse_s + mse_H + mse_smb + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s, "mse_H": mse_H, "mse_smb": mse_smb,
                "mse_fSSA1": mse_fSSA1, "mse_fSSA2": mse_fSSA2, "mse_fH": mse_fH, "mse_fc1": mse_fc1, "mse_fc2": mse_fc2} 

    def predict(self, X_star):
        '''
        return numpy array of the model
        '''
        sol_pred = self.model(X_star)
        u_pred = sol_pred[:, 0:1]
        v_pred = sol_pred[:, 1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:, 0:1]
        H_pred = sH_pred[:, 1:2]

        C_pred = self.C_model(X_star[:,0:2])

        smb_pred = self.smb_model(X_star)

        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), smb_pred.numpy()

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
    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.smb_model.save(self.modelPath+"/smb_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA2D_frictionNN_uvsH(SSA2D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath+"/model/", reloadModel,
                mu, loss_weights=loss_weights)
        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
            self.friction_model = tf.keras.models.load_model(modelPath+"/friction_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

            # hp["friction_layers"] defines friction model
            fri_lb = (ulb[4:5]**2)*((ulb[0:1]**2.0+ulb[1:2]**2.0)**(0.5/n))
            fri_ub = (uub[4:5]**2)*((uub[0:1]**2.0+uub[1:2]**2.0)**(0.5/n))
            self.friction_model = create_NN(hp["friction_layers"], inputRange=(ulb[0:4], uub[0:4]), outputRange=(fri_lb, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, 0

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

            # use NN to predict the basal stress
            tempX = tf.concat([u, v, s, H], axis=1)
            taub = self.friction_model(tempX) 

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

        f1 = sigma11 + sigma12 - taub*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - taub*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:,0:1]
        v_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:,0:1]
        H_pred = sH_pred[:,1:2]

        uvsH_pred = tf.concat([u_pred, v_pred, s_pred, H_pred], axis=1)
        sol_pred = self.friction_model(uvsH_pred)

        # ref taub
        ref_sol = u_star[:,4:5]**2*((u_star[:,0:1]**2.0+u_star[:,1:2]**2.0)**(1.0/6.0))

        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)

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

        
        uvsH_pred = tf.concat([u_pred, v_pred, s_pred, H_pred], axis=1)
        taub_pred = self.friction_model(uvsH_pred)
        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), taub_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()
    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.friction_model.save(self.modelPath+"/friction_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA2D_frictionNN_uvsH_positiveTau(SSA2D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath+"/model/", reloadModel,
                mu, loss_weights=loss_weights)
        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
            self.friction_model = tf.keras.models.load_model(modelPath+"/friction_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

            # hp["friction_layers"] defines friction model
            fri_lb = ((ulb[4:5]**2)*((ulb[0:1]**2.0+ulb[1:2]**2.0)**(0.5/n)) )**0.5
            fri_ub = ((uub[4:5]**2)*((uub[0:1]**2.0+uub[1:2]**2.0)**(0.5/n)) )**0.5
            self.friction_model = create_NN(hp["friction_layers"], inputRange=(ulb[0:4], uub[0:4]), outputRange=(fri_lb, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, 0

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

            # use NN to predict the basal stress
            tempX = tf.concat([u, v, s, H], axis=1)
            taub = self.friction_model(tempX) 

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

        f1 = sigma11 + sigma12 - (taub*taub)*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - (taub*taub)*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:,0:1]
        v_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:,0:1]
        H_pred = sH_pred[:,1:2]

        uvsH_pred = tf.concat([u_pred, v_pred, s_pred, H_pred], axis=1)
        sol_pred = self.friction_model(uvsH_pred)

        # ref taub
        ref_sol = u_star[:,4:5]**2*((u_star[:,0:1]**2.0+u_star[:,1:2]**2.0)**(1.0/6.0))

        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.square(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)

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

        
        uvsH_pred = tf.concat([u_pred, v_pred, s_pred, H_pred], axis=1)
        taub_pred = (self.friction_model(uvsH_pred))**2
        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), taub_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()
    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.friction_model.save(self.modelPath+"/friction_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA2D_frictionNN_uvsH_positiveTau_velmag(SSA2D): #{{{
    '''
    class of learning friction laws from observed u, H, and h, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath+"/model/", reloadModel,
                mu, loss_weights=loss_weights)
        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
            self.friction_model = tf.keras.models.load_model(modelPath+"/friction_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

            # hp["friction_layers"] defines friction model
            maxvel = (max(ulb[0]**2,uub[0]**2)+max(ulb[1]**2,uub[1]**2))**0.5;
            fri_ub = ((uub[4:5]**2)*(maxvel**(1.0/n)) )**0.5
            self.friction_model = create_NN(hp["friction_layers"], inputRange=(np.concatenate([[0.0],ulb[2:4]]), np.concatenate([[maxvel], uub[2:4]])), outputRange=(-fri_ub, fri_ub))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.friction_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.friction_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y, s, H, 0

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

            u_norm = (u**2+v**2)**0.5
            # use NN to predict the basal stress
            tempX = tf.concat([u_norm, s, H], axis=1)
            taub = self.friction_model(tempX) 

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

        f1 = sigma11 + sigma12 - (taub*taub)*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - (taub*taub)*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return f1, f2

    @tf.function
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]
        H0 = uv[:,3:4]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]
        H0_pred = sH_pred[:,1:2]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # velocity misfit
        mse_u = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = self.loss_weights[0]*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        # geometry misfit
        mse_s = self.loss_weights[1]*tf.reduce_mean(tf.square(s0 - s0_pred))
        mse_H = self.loss_weights[1]*tf.reduce_mean(tf.square(H0 - H0_pred))
        # residual of PDE
        mse_f1 = self.loss_weights[3]*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = self.loss_weights[3]*tf.reduce_mean(tf.square(f2_pred))
        # calving front boundary
        mse_fc1 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = self.loss_weights[4]*tf.reduce_mean(tf.square(fc2_pred))

        # sum the total
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        uv_pred = self.model(X_star)
        u_pred = uv_pred[:,0:1]
        v_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_star)
        s_pred = sH_pred[:,0:1]
        H_pred = sH_pred[:,1:2]

        u_norm_pred = tf.math.sqrt(tf.math.square(u_pred)+tf.math.square(v_pred))
        uvsH_pred = tf.concat([u_norm_pred, s_pred, H_pred], axis=1)
        sol_pred = self.friction_model(uvsH_pred)

        # ref taub
        ref_sol = u_star[:,4:5]**2*((u_star[:,0:1]**2.0+u_star[:,1:2]**2.0)**(1.0/6.0))

        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.square(ref_sol)) / tf.math.reduce_euclidean_norm(ref_sol)

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

        u_norm_pred = tf.math.sqrt(tf.math.square(u_pred)+tf.math.square(v_pred))
        
        uvsH_pred = tf.concat([u_norm_pred, s_pred, H_pred], axis=1)
        taub_pred = (self.friction_model(uvsH_pred))**2
        return u_pred.numpy(), v_pred.numpy(), s_pred.numpy(), H_pred.numpy(), C_pred.numpy(), taub_pred.numpy()

    def summary(self):
        '''
        output all model summaries
        '''
        return self.model.summary(),self.h_model.summary(), self.C_model.summary(), self.friction_model.summary()
    def save(self):
        '''
        save the model and history of training
        '''
        self.model.save(self.modelPath+"/model")
        self.h_model.save(self.modelPath+"/h_model")
        self.C_model.save(self.modelPath+"/C_model")
        self.friction_model.save(self.modelPath+"/friction_model")
        self.logger.save(self.modelPath+"/history.json")
    #}}}
class SSA2D_3NN_invertHandC(SSA2D): #{{{
    '''
    class of learning C and H from observed u, v, and s, and PDEs
    '''
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            mu, n=3.0, 
            loss_weights=[1e-5, 1e-3, 1e-5, 1e-8, 1e-12]):
        super().__init__(hp, logger, X_f, 
                X_bc, u_bc, X_cf, n_cf,
                xub, xlb, uub[0:2], ulb[0:2],
                modelPath+"/model/", reloadModel,
                mu, loss_weights=loss_weights)
        # overwrite self.modelPath, which has been changed in super()
        self.modelPath = modelPath
        if reloadModel and os.path.exists(self.modelPath):
            #load
            self.h_model = tf.keras.models.load_model(modelPath+"/h_model/")
            self.C_model = tf.keras.models.load_model(modelPath+"/C_model/")
        else:
            # hp["h_layers"] defines h and H model
            self.h_model = create_NN(hp["h_layers"], inputRange=(xlb, xub), outputRange=(ulb[2:4], uub[2:4]))

            # hp["C_layers"] defines C model
            self.C_model = create_NN(hp["C_layers"], inputRange=(xlb, xub), outputRange=(ulb[4:5], uub[4:5]))

        self.trainableLayers = (self.model.layers[1:-1]) + (self.h_model.layers[1:-1]) + (self.C_model.layers[1:-1])
        self.trainableVariables = self.model.trainable_variables + self.h_model.trainable_variables + self.C_model.trainable_variables

    # need to overwrite nn_model, which is used in computing the loss function
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
    def loss(self, uv, X_u):
        '''
        loss = |u-uobs|+|v-vobs|+|h-hobs|+|H-Hobs|+|f1|+|f2|
        '''
        # Dirichlet B.C. for H and C
        H0 = self.u_bc[:, 3:4]
        sH_bc = self.h_model(self.X_bc)
        H0_pred = sH_bc[:,1:2]

        C0 = self.u_bc[:, 4:5]
        C0_pred = self.C_model(self.X_bc)

        # match h, H, and C to the training data
        u0 = uv[:,0:1]
        v0 = uv[:,1:2]
        s0 = uv[:,2:3]

        uv_pred = self.model(X_u)
        u0_pred = uv_pred[:,0:1]
        v0_pred = uv_pred[:,1:2]

        sH_pred = self.h_model(X_u)
        s0_pred = sH_pred[:,0:1]

        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()
        # calving front
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
        totalloss = mse_u + mse_v + mse_s + mse_H + mse_C + mse_f1 + mse_f2 + mse_fc1 + mse_fc2
        return {"loss": totalloss, "mse_u": mse_u, "mse_v": mse_v, "mse_s": mse_s,
                "mse_H": mse_H, "mse_C": mse_C, "mse_f1": mse_f1, "mse_f2": mse_f2,
                "mse_fc1": mse_fc1, "mse_fc2": mse_fc2}

    @tf.function
    def test_error(self, X_star, u_star):
        '''
        test error of taub
        '''
        sol_pred = self.C_model(X_star)
        return  tf.math.reduce_euclidean_norm(tf.math.abs(sol_pred) - tf.math.abs(u_star[:,4:5])) / tf.math.reduce_euclidean_norm(u_star[:,4:5])

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
