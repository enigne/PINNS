import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import math

yts = 3600*24*365

def plot_SSA(pinn, X_f, X_star, u_star, xlb, xub): #{{{
    u_pred, v_pred = pinn.predict(X_star)

    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')


    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs u')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)

    ax = axs[0][1]
    im = ax.imshow(uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs v')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ################################
    ax = axs[1][0]
    rg = max(abs(u_nn-ux))
    im = ax.imshow(u_nn-ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin = -rg, vmax=rg, 
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(v_nn-uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs v')
    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[2][0]
    im = ax.imshow(F1, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('f1 residual')
    fig.colorbar(im, ax=ax, shrink=1)
    ax = axs[2][1]
    im = ax.imshow(F2, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('f2 residual')
    fig.colorbar(im, ax=ax, shrink=1)

    plt.show()
    #}}}
def plot_SSA_C(pinn, X_star, u_star, xlb, xub): #{{{
    u_pred, v_pred = pinn.predict(X_star)
    h_pred = pinn.C_model(X_star)
    C_pred = (h_pred[:, 0:1]).numpy()

    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    C = griddata(X_star, u_star[:,2].flatten(), (X, Y), method='cubic')
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0].flatten(), (X, Y), method='cubic')


    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs u')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)


    ax = axs[0][1]
    im = ax.imshow(uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs v')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ax = axs[0][2]
    im = ax.imshow(C, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=0.5, vmax=1.5,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('C')
    fig.colorbar(im, ax=ax, shrink=1)

    ################################
    ax = axs[1][0]
    im = ax.imshow(u_nn - ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(v_nn - uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs v')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[1][2]
    im = ax.imshow(C_nn - C, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs C')
    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[2][0]
    im = ax.imshow(F1, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('f1 residual')
    fig.colorbar(im, ax=ax, shrink=1)
    ax = axs[2][1]
    im = ax.imshow(F2, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('f2 residual')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[2][2]
    im = ax.imshow(C_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=0.5, vmax=1.5,
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict C')
    fig.colorbar(im, ax=ax, shrink=1)

    plt.show()
    #}}}
def plot_H_bed(pinn, X_star, hb_star, xlb, xub): #{{{
    H_pred, b_pred = pinn.geometry_NN(X_star)
    H, b = pinn.geometry_model(X_star)

    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    H = griddata(X_star, (H[:,0]).flatten(), (X, Y), method='cubic')
    b = griddata(X_star, (b[:,0]).flatten(), (X, Y), method='cubic')
    H_nn = griddata(X_star, (H_pred[:,0].numpy()).flatten(), (X, Y), method='cubic')
    b_nn = griddata(X_star, (b_pred[:,0].numpy()).flatten(), (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(H, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs H')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)

    ax = axs[0][1]
    im = ax.imshow(b, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs bed')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ################################
    ax = axs[1][0]
    im = ax.imshow(H_nn - H, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs H')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(b_nn - b, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs bed')
    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[2][0]
    im = ax.imshow(H_nn, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict H')
    fig.colorbar(im, ax=ax, shrink=1)
    ax = axs[2][1]
    im = ax.imshow(b_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict bed')
    fig.colorbar(im, ax=ax, shrink=1)

    plt.show()
    #}}}
def plot_H_bed_train(pinn, X_star, u_star, xlb, xub): #{{{
    H_pred, b_pred = pinn.predict(X_star)

    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    H = griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    b = griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    H_nn = griddata(X_star, H_pred[:,0].flatten(), (X, Y), method='cubic')
    b_nn = griddata(X_star, b_pred[:,0].flatten(), (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(H, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs H')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)

    ax = axs[0][1]
    im = ax.imshow(b, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs bed')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ################################
    ax = axs[1][0]
    im = ax.imshow(H_nn - H, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs H')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(b_nn - b, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs bed')
    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[2][0]
    im = ax.imshow(H_nn, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict H')
    fig.colorbar(im, ax=ax, shrink=1)
    ax = axs[2][1]
    im = ax.imshow(b_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict bed')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[2][2]
    ax.plot((pinn.logger.loss_history), label="loss")
    ax.plot((pinn.logger.test_history), label="test")
    ax.axes.set_yscale('log')
    plt.legend()

    plt.show()
    #}}}
def plot_C_train(pinn, X_star, u_star, xlb, xub): #{{{
    C_pred = pinn.predict(X_star)

    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    C = griddata(X_star, u_star[:,None].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0].flatten(), (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(C, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs C')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)

    ################################
    ax = axs[1][0]
    im = ax.imshow(C_nn - C, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs C')
    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[2][0]
    im = ax.imshow(C_nn, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict C')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[2][1]
    ax.plot((pinn.logger.history["loss"]), label="loss")
    ax.plot((pinn.logger.history["test"]), label="test")
    ax.axes.set_yscale('log')
    plt.legend()

    plt.show()
    #}}}
def plot_Helheim(pinn, X_f, X_star, u_star, xlb, xub): #{{{
    u_pred, v_pred = pinn.predict(X_star)
    C  = pinn.C_model(X_star)
    C_pred = C[:, 0].numpy()
    hb_pred = pinn.H_bed_model(X_star)
    H = hb_pred[:,0:1]
    b = hb_pred[:,1:2]
#    hx = hb_pred[:,2:3]
#    hy = hb_pred[:,3:4]
#    hxy = (hx**2+hy**2)**0.5

    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred, (X, Y), method='cubic')
    H_nn = griddata(X_star, H, (X, Y), method='cubic')
    b_nn = griddata(X_star, b, (X, Y), method='cubic')
 #   hxy_nn = griddata(X_star, hxy, (X, Y), method='cubic')

    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 4, figsize=(16,12))

    ax = axs[0][0]
    im = ax.imshow(ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs u')
    fig.colorbar(im, ax=ax, shrink=1)
#    ax.plot(X_star[:,0],X_star[:,1], 'k*', markersize = 2, clip_on = False)
    ax.plot(pinn.X_cf[:,0], pinn.X_cf[:,1], 'k*', markersize = 2, clip_on = False)

    ax = axs[0][1]
    im = ax.imshow(uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs v')
    fig.colorbar(im, ax=ax, shrink=1)
    ax.plot(pinn.X_bc[:,0], pinn.X_bc[:,1], 'k*', markersize = 2, clip_on = False)
 #   ax.plot(X_f[:,0],X_f[:,1], 'k*', markersize = 2, clip_on = False)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ax = axs[0][2]
    im = ax.imshow(C_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('friction C')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ax = axs[0][3]
    im = ax.imshow(H_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('Thickness')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*',  markersize = 2, clip_on = False)

    ################################
    ax = axs[1][0]
    im = ax.imshow(u_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(v_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict v')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[1][3]
    im = ax.imshow(b_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('bed')
    fig.colorbar(im, ax=ax, shrink=1)

    ################################
    ax = axs[2][0]
    im = ax.imshow(u_nn - ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            #vmin = -400, vmax=400,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[2][1]
    im = ax.imshow(v_nn - uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            # vmin = -100, vmax=100,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs v')
    fig.colorbar(im, ax=ax, shrink=1)

#    ax = axs[2][3]
#    im = ax.imshow(hxy_nn, interpolation='nearest', cmap='rainbow',
#            extent=[X.min(), X.max(), Y.min(), Y.max()],
#            origin='lower', aspect='auto')
#    # ax.set_xlabel('x')
#    # ax.set_ylabel('y')
#    ax.set_title('surface gradient')
#    fig.colorbar(im, ax=ax, shrink=1)

    #########################################
    ax = axs[1][2]
    im = ax.imshow(F1, interpolation='none', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('f1 residual')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[2][2]
    im = ax.imshow(F2, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('f2 residual')
    fig.colorbar(im, ax=ax, shrink=1)

    ax = axs[2][3]
    ax.plot((pinn.logger.loss_history), label="loss")
    ax.plot((pinn.logger.test_history), label="test")
    ax.axes.set_yscale('log')
    plt.legend()

    plt.show()
    #}}}
def plot_Helheim_all(pinn, X_f, X_star, u_star, xlb, xub, vranges={}): #{{{
    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    # obs
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    h_obs = griddata(X_star, u_star[:,2].flatten(), (X, Y), method='cubic')
    H_obs = griddata(X_star, u_star[:,3].flatten(), (X, Y), method='cubic')
    C_obs = griddata(X_star, u_star[:,4].flatten(), (X, Y), method='cubic')

    # predicted solution
    u_pred, v_pred, h, H, C_pred = pinn.predict(X_star)
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0], (X, Y), method='cubic')
    h_nn = griddata(X_star, h[:,0], (X, Y), method='cubic')
    H_nn = griddata(X_star, H[:,0], (X, Y), method='cubic')

    # residual
    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    ###########################
    plotData = {}
    plotData['u obs'] = ux
    plotData['v obs'] = uy
    plotData['C - C obs'] = abs(C_nn) - abs(C_obs)
    plotData['h - h obs'] = h_nn - h_obs
    ###########################
    plotData['u pred'] = u_nn
    plotData['v pred'] = v_nn
    plotData['C pred'] = abs(C_nn)
    plotData['H - H obs'] = H_nn - H_obs
    ###########################
    plotData['u - u obs'] = u_nn - ux
    plotData['v - v obs'] = v_nn - uy
    plotData['f1 residual'] = F1
    plotData['f2 residual'] = F2

    fig, axs = plt.subplots(3, 4, figsize=(16,12))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        vr = vranges.setdefault(name, [None, None])
        im = ax.imshow(plotData[name], interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                vmin=vr[0], vmax=vr[1],
                origin='lower', aspect='auto')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=1)

    #ax.plot(pinn.X_bc[:,0], pinn.X_bc[:,1], 'k*', markersize = 2, clip_on = False)
    plt.show()
    #}}}
def plot_log_history(pinn, savePath=None): #{{{

    n = len(pinn.logger.history.keys())
    cols = 4

    fig, axs = plt.subplots(math.ceil(n/cols), cols, figsize=(16,12))

    for ax, name in zip(axs.ravel(), pinn.logger.history.keys()):
        ax.plot((pinn.logger.history[name]), label=name)
        ax.axes.set_yscale('log')
        ax.legend(loc="best")

    if savePath:
        plt.savefig(savePath+"/history.png")
    else:
        plt.show()
    #}}}
def plot_2D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, vranges={}, savePath=None): #{{{
    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    # obs
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    h_obs = griddata(X_star, u_star[:,2].flatten(), (X, Y), method='cubic')
    H_obs = griddata(X_star, u_star[:,3].flatten(), (X, Y), method='cubic')
    C_obs = griddata(X_star, u_star[:,4].flatten(), (X, Y), method='cubic')

    # predicted solution
    u_pred, v_pred, h, H, C_pred = pinn.predict(X_star)
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0], (X, Y), method='cubic')
    h_nn = griddata(X_star, h[:,0], (X, Y), method='cubic')
    H_nn = griddata(X_star, H[:,0], (X, Y), method='cubic')

    # residual
    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    ###########################
    plotData = {}
    plotData['u obs'] = ux
    plotData['v obs'] = uy
    plotData['C - C obs'] = abs(C_nn) - abs(C_obs)
    plotData['h - h obs'] = h_nn - h_obs
    ###########################
    plotData['u pred'] = u_nn
    plotData['v pred'] = v_nn
    plotData['C pred'] = abs(C_nn)
    plotData['H - H obs'] = H_nn - H_obs
    ###########################
    plotData['u - u obs'] = u_nn - ux
    plotData['v - v obs'] = v_nn - uy
    plotData['f1 residual'] = F1
    plotData['f2 residual'] = F2

    fig, axs = plt.subplots(3, 4, figsize=(16,12))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        vr = vranges.setdefault(name, [None, None])
        im = ax.imshow(plotData[name], interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                vmin=vr[0], vmax=vr[1],
                origin='lower', aspect='auto')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=1)

    if savePath:
        plt.savefig(savePath+"/2Dsolution.png")
    else:
        plt.show()
    #}}}
def plot_1D_solutions_all(pinn, X_f, X_star, u_star, xlb, xub, vranges={}, savePath=None): #{{{
    yts = 3600*24*365
    # ref solution
    u_obs = u_star[:,0:1] * yts
    h_obs = u_star[:,1:2]
    H_obs = u_star[:,2:3]
    C_obs = u_star[:,3:4]

    # predicted solution, only look at X_star
    u_pred, h_nn, H_nn, C_nn = pinn.predict(X_star)
    u_nn = yts * u_pred

    # residual
    f1 = pinn.f_model()

    ###########################
    plotData = {}
    plotData['u - u obs'] = u_nn - u_obs
    plotData['h - h obs'] = h_nn - h_obs
    plotData['H - H obs'] = H_nn - H_obs
    ###########################
    plotData['u'] = np.hstack((u_nn, u_obs))
    plotData['h'] = np.hstack((h_nn, h_obs))
    plotData['H'] = np.hstack((H_nn, H_obs))
    ###########################
    plotData['f1 residual'] = f1
    plotData['C - C obs'] = abs(C_nn) - abs(C_obs)
    plotData['C'] = np.hstack([abs(C_nn), abs(C_obs)])

    fig, axs = plt.subplots(3, 3, figsize=(16,12))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        if 'f1' in name:
            im = ax.plot(X_f, plotData[name], 'o')
        else:
            im = ax.plot(X_star, plotData[name])
        ax.set_title(name)

    if savePath:
        plt.savefig(savePath+"/1Dsolution.png")
    else:
        plt.show()
    #}}}
def plot_1D_transient_solutions(pinn, X_f, X_star, u_star, X_1d, C_1d, xlb, xub, vranges={}, savePath=None): #{{{
    yts = 3600*24*365
    # predicted C solution, only look at X_1d
    C_nn = pinn.C_model(X_1d)
    plotData1D = {}
    plotData1D['C - C obs'] = abs(C_nn) - abs(C_1d)
    plotData1D['C and C obs'] = np.hstack([abs(C_nn), abs(C_1d)])
    fig, axs = plt.subplots(1, 3, figsize=(16,5))

    for ax,name in zip(axs.ravel(), plotData1D.keys()):
        if 'f1' in name:
            im = ax.plot(X_1d, plotData1D[name], 'o')
        else:
            im = ax.plot(X_1d, plotData1D[name])
        ax.set_title(name)
    if savePath:
        plt.savefig(savePath+"/1Dsolution.png")
    else:
        plt.show()
        
    # 2D solutions x-t plane
    X, T = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    # obs
    u = yts*griddata(X_star, u_star[:,0].flatten(), (X, T), method='cubic')
    h_obs = griddata(X_star, u_star[:,1].flatten(), (X, T), method='cubic')
    H_obs = griddata(X_star, u_star[:,2].flatten(), (X, T), method='cubic')
    smb_obs = yts*griddata(X_star, u_star[:,3].flatten(), (X, T), method='cubic')

    # predicted solution
    u_pred, h, H, C_pred, smb = pinn.predict(X_star)
    smb = pinn.smb_model(X_star)
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, T), method='cubic')
    h_nn = griddata(X_star, h[:,0], (X, T), method='cubic')
    H_nn = griddata(X_star, H[:,0], (X, T), method='cubic')
    smb_nn = yts*griddata(X_star, smb[:,0], (X, T), method='cubic')

    # residual
    fSSA, fH = pinn.f_model()
    FSSA = griddata(X_f, fSSA[:,0], (X, T), method='cubic')
    FH = yts*griddata(X_f, fH[:,0], (X, T), method='cubic')
    
    ###########################
    plotData = {}
    plotData['u obs'] = u
    plotData['h obs'] = h_obs
    plotData['H obs'] = H_obs
    plotData['smb obs'] = smb_obs
    plotData['fSSA residual'] = FSSA
    ###########################
    plotData['u nn'] = u_nn
    plotData['h nn'] = h_nn
    plotData['H nn'] = H_nn
    plotData['smb nn'] = smb_nn
    plotData['fH residual'] = FH
    ###########################
    plotData['u - u obs'] = u_nn - u
    plotData['h - h obs'] = h_nn - h_obs
    plotData['H - H obs'] = H_nn - H_obs
    plotData['smb - smb obs'] = smb_nn - smb_obs

    fig, axs = plt.subplots(3, 5, figsize=(24,12))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        vr = vranges.setdefault(name, [None, None])
        im = ax.imshow(plotData[name], interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), T.min()/yts, T.max()/yts],
                vmin=vr[0], vmax=vr[1],
                origin='lower', aspect='auto')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=1)

    if savePath:
        plt.savefig(savePath+"/2Dsolution.png")
    else:
        plt.show()
    #}}}
def plot_2D_frictionNN(pinn, X_f, X_star, u_star, xlb, xub, vranges={}, savePath=None): #{{{
    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    # obs
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    h_obs = griddata(X_star, u_star[:,2].flatten(), (X, Y), method='cubic')
    H_obs = griddata(X_star, u_star[:,3].flatten(), (X, Y), method='cubic')
    C_obs = griddata(X_star, u_star[:,4].flatten(), (X, Y), method='cubic')
    taub_comp = u_star[:,4:5]**2*((u_star[:,0:1]**2.0+u_star[:,1:2]**2.0)**(1.0/6.0))
    taub_model = griddata(X_star, taub_comp[:,0], (X, Y), method='cubic')

    # predicted solution
    u_pred, v_pred, h, H, C_pred, taub = pinn.predict(X_star)
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0], (X, Y), method='cubic')
    h_nn = griddata(X_star, h[:,0], (X, Y), method='cubic')
    H_nn = griddata(X_star, H[:,0], (X, Y), method='cubic')
    taub_nn = griddata(X_star, taub[:,0], (X, Y), method='cubic')

    # residual
    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    ###########################
    plotData = {}
    plotData['u obs'] = ux
    plotData['v obs'] = uy
    plotData['taub from ISSM C'] = taub_model
    plotData['h - h obs'] = h_nn - h_obs
    ###########################
    plotData['u pred'] = u_nn
    plotData['v pred'] = v_nn
    plotData['taub pred'] = taub_nn
    plotData['H - H obs'] = H_nn - H_obs
    ###########################
    plotData['u - u obs'] = u_nn - ux
    plotData['v - v obs'] = v_nn - uy
    plotData['taub - taub obs'] = taub_nn - taub_model
    plotData['f1 residual'] = F1
    plotData['f2 residual'] = F2

    fig, axs = plt.subplots(4, 4, figsize=(16,16))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        vr = vranges.setdefault(name, [None, None])
        im = ax.imshow(plotData[name], interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                vmin=vr[0], vmax=vr[1],
                origin='lower', aspect='auto')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=1)

    if savePath:
        plt.savefig(savePath+"/2Dsolution.png")
    else:
        plt.show()
    #}}}
def plot_2D_solutions_mu(pinn, X_f, X_star, u_star, xlb, xub, vranges={}, savePath=None): #{{{
    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0],200), np.linspace(xlb[1],xub[1], 200))
    # obs
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    h_obs = griddata(X_star, u_star[:,2].flatten(), (X, Y), method='cubic')
    H_obs = griddata(X_star, u_star[:,3].flatten(), (X, Y), method='cubic')
    C_obs = griddata(X_star, u_star[:,4].flatten(), (X, Y), method='cubic')

    # predicted solution
    u_pred, v_pred, h, H, C_pred, mu_pred = pinn.predict(X_star)
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred[:,0], (X, Y), method='cubic')
    h_nn = griddata(X_star, h[:,0], (X, Y), method='cubic')
    H_nn = griddata(X_star, H[:,0], (X, Y), method='cubic')

    # residual
    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    ###########################
    plotData = {}
    plotData['u obs'] = ux
    plotData['v obs'] = uy
    plotData['C - C obs'] = abs(C_nn) - abs(C_obs)
    plotData['h - h obs'] = h_nn - h_obs
    ###########################
    plotData['u pred'] = u_nn
    plotData['v pred'] = v_nn
    plotData['C pred'] = abs(C_nn)
    plotData['H - H obs'] = H_nn - H_obs
    ###########################
    plotData['u - u obs'] = u_nn - ux
    plotData['v - v obs'] = v_nn - uy
    plotData['f1 residual'] = F1
    plotData['f2 residual'] = F2

    fig, axs = plt.subplots(3, 4, figsize=(16,12))

    for ax,name in zip(axs.ravel(), plotData.keys()):
        vr = vranges.setdefault(name, [None, None])
        im = ax.imshow(plotData[name], interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                vmin=vr[0], vmax=vr[1],
                origin='lower', aspect='auto')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=1)

    if savePath:
        plt.savefig(savePath+"/2Dsolution.png")
    else:
        plt.show()
    #}}}
