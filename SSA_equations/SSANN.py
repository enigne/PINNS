import sys
import os
import tensorflow as tf
import numpy as np
sys.path.append("../utils")
from custom_lbfgs import *
from neuralnetwork import NeuralNetwork, MinmaxScaleLayer, UpScaleLayer
from logger import Logger

class SSAInformedNN(NeuralNetwork): #{{{
    def __init__(self, hp, logger, X_f, 
            X_bc, u_bc, X_cf, n_cf, 
            xub, xlb, uub, ulb, 
            modelPath, reloadModel,
            eta, n=3.0, geoDataNN=None, FrictionCNN=None):
        super().__init__(hp, logger, xub, xlb, uub[0:2], ulb[0:2], modelPath, reloadModel=reloadModel)

        # friction C model
        if FrictionCNN:
            self.C_model = tf.keras.models.load_model(FrictionCNN)
        else:
            # set C model
            C_layers = hp["C_layers"]
            self.C_model = tf.keras.Sequential()
            # input layer
            self.C_model.add(tf.keras.layers.InputLayer(input_shape=(C_layers[0],)))
            # normalization layer
            self.C_model.add(MinmaxScaleLayer(xlb, xub))
            # NN layers
            for width in C_layers[1:-1]:
                self.C_model.add(tf.keras.layers.Dense(
                    width, activation=tf.nn.tanh,
                    kernel_initializer="glorot_normal"))
            # output layer
            self.C_model.add(tf.keras.layers.Dense(
                    C_layers[-1], activation=None,
                    kernel_initializer="glorot_normal"))
    
            # denormalization layer
            self.C_model.add(UpScaleLayer(ulb[2], uub[2]))
    
            # Computing the sizes of weights/biases for future decomposition
            for i, width in enumerate(C_layers):
                if i > 0:
                    self.sizes_w.append(int(width * C_layers[i-1]))
                    self.sizes_b.append(int(width))
                
            # set up trainable layers and variables
            self.trainableLayers = (self.model.layers[1:-1]) + (self.C_model.layer[1:-1])
            self.trainableVariables = self.model.trainable_variables + self.C_model.trainable_variables

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
        self.eta = tf.constant(eta, dtype=self.dtype)
        self.n = tf.constant(n, dtype=self.dtype)

        # some constants
        self.rhoi = tf.constant(917, dtype=self.dtype)  # kg/m^3
        self.rhow = tf.constant(1023, dtype=self.dtype) # kg/m^3
        self.g = tf.constant(9.81, dtype=self.dtype)    # m/s^2
        self.yts = tf.constant(3600.0*24*365, dtype=self.dtype)

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])

        # use H and bed interpolator
        if geoDataNN: 
            self.H_bed_model = tf.keras.models.load_model(geoDataNN)            

    def geometry_NN(self, X):
        Hb = self.H_bed_model(X)
        H = Hb[:, 0:1]
        b = Hb[:, 1:2]
        return H, b

    # get the velocity and derivative information
    @tf.function
    def uvx_model(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            UV = self.model(Xtemp)
            u = UV[:, 0:1]
            v = UV[:, 1:2]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        u_y = tape.gradient(u, y)
        v_y = tape.gradient(v, y)
        del tape

        return u, v, u_x, v_x, u_y, v_y

    # Calving front condition
    @tf.function
    def cf_model (self, X, n):
        x = X[:, 0:1]
        y = X[:, 1:2]
        nx = n[:, 0:1]
        ny = n[:, 1:2]

        # viscosity
        eta = self.eta
        n = self.n

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([x, y], axis=1)

            # geometry
            H, bed = self.geometry_NN(Xtemp)
            h = H + bed

            # velocity component
            u, v, u_x, v_x, u_y, v_y = self.uvx_model(Xtemp)

            # viscosity
            epsilon = 0.5*eta *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = epsilon * H
            B11 = etaH*(4*u_x + 2*v_y)
            B22 = etaH*(4*v_y + 2*u_x)
            B12 = etaH*(  u_y +   v_x)
        del tape

        # Calving front condition
        fc1 = B11 *nx + B12*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*bed*bed)*nx
        fc2 = B12 *nx + B22*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*bed*bed)*ny

        return fc1, fc2

    # The actual PINN
    @tf.function
    def f_model(self):
        # viscosity
        eta = self.eta
        n = self.n

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and y
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            # Packing together the inputs
            X_f = tf.concat([self.x_f, self.y_f], axis=1)

            # get ice thickness and bed
            #H, bed, h_x, h_y = self.geometry_NN(X_f)
            H, bed = self.geometry_NN(X_f)
            h = H + bed

            # Getting the prediction
            u, v, u_x, v_x, u_y, v_y = self.uvx_model(X_f)

            epsilon = 0.5*eta *(u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y+1.0e-30)**(0.5*(1.0-n)/n)
            # stress tensor
            etaH = epsilon * H
            B11 = etaH*(4*u_x + 2*v_y)
            B22 = etaH*(4*v_y + 2*u_x)
            B12 = etaH*(  u_y +   v_x)

            # friction
            C = self.C_model(X_f)

        # Getting the other derivatives
        sigma11 = tape.gradient(B11, self.x_f)
        sigma12 = tape.gradient(B12, self.y_f)

        sigma21 = tape.gradient(B12, self.x_f)
        sigma22 = tape.gradient(B22, self.y_f)
                
        # surface gradient
        h_x = tape.gradient(h, self.x_f)
        h_y = tape.gradient(h, self.y_f)

        # Letting the tape go
        del tape
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*h_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*h_y

        return f1, f2

    @tf.function
    def loss(self, uv, uv_pred):
        # Dirichlet conditions from training
        u0 = uv[:, 0:1]
        v0 = uv[:, 1:2]
        u0_pred = uv_pred[:, 0:1]
        v0_pred = uv_pred[:, 1:2]
        
        # f_model on the collocation points 
        f1_pred, f2_pred = self.f_model()

        # Calving on X_cf
        fc1_pred, fc2_pred = self.cf_model(self.X_cf, self.n_cf)

        # misfits
        mse_u = 1e-2*(self.yts**2) * tf.reduce_mean(tf.square(u0 - u0_pred))
        mse_v = 1e-2*(self.yts**2) * tf.reduce_mean(tf.square(v0 - v0_pred))
        mse_f1 = 1e-6*tf.reduce_mean(tf.square(f1_pred))
        mse_f2 = 1e-6*tf.reduce_mean(tf.square(f2_pred))
        mse_fc1 = 1e-10*tf.reduce_mean(tf.square(fc1_pred))
        mse_fc2 = 1e-10*tf.reduce_mean(tf.square(fc2_pred))

        return mse_u + mse_v + \
                mse_f1 + mse_f2 + \
                mse_fc1 + mse_fc2

    def predict(self, X_star):
        h_pred = self.model(X_star)
        u_pred = h_pred[:, 0:1]
        v_pred = h_pred[:, 1:2]
        return u_pred.numpy(), v_pred.numpy()

    @tf.function
    def test_error(self, X_star, u_star):
        h_pred = self.model(X_star)
        return  tf.math.reduce_euclidean_norm(h_pred - u_star[:,0:2]) / tf.math.reduce_euclidean_norm(u_star[:,0:2])
    #}}}