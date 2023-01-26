import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

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
    im = ax.imshow(u_nn-ux, interpolation='nearest', cmap='rainbow',
                extent=[X.min(), X.max(), Y.min(), Y.max()],
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

    plt.show()
    #}}}
def plot_Helheim(pinn, X_f, X_star, u_star, xlb, xub): #{{{
    u_pred, v_pred = pinn.predict(X_star)
    C  = pinn.C_model(X_star)
    C_pred = C[:, 0].numpy()

    yts = 3600*24*365
    X, Y = np.meshgrid(np.linspace(xlb[0],xub[0]), np.linspace(xlb[1],xub[1]))
    ux = yts*griddata(X_star, u_star[:,0].flatten(), (X, Y), method='cubic')
    uy = yts*griddata(X_star, u_star[:,1].flatten(), (X, Y), method='cubic')
    u_nn = yts*griddata(X_star, u_pred[:,0].flatten(), (X, Y), method='cubic')
    v_nn = yts*griddata(X_star, v_pred[:,0].flatten(), (X, Y), method='cubic')
    C_nn = griddata(X_star, C_pred, (X, Y), method='cubic')

    f1, f2 = pinn.f_model()
    F1 = griddata(X_f, f1[:,0], (X, Y), method='cubic')
    F2 = griddata(X_f, f2[:,0], (X, Y), method='cubic')

    fig, axs = plt.subplots(3, 3, figsize=(12,12))

    ax = axs[0][0]
    im = ax.imshow(ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=-100, vmax=200,
            origin='lower', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('obs u')
    fig.colorbar(im, ax=ax, shrink=1)
    # ax.plot(X_u_train[:,0],X_u_train[:,1], 'k*', markersize = 2, clip_on = False)

    ax = axs[0][1]
    im = ax.imshow(uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=0, vmax=1200,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('obs v')
    fig.colorbar(im, ax=ax, shrink=1)
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

    ################################
    ax = axs[1][0]
    im = ax.imshow(u_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=-100, vmax=200,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[1][1]
    im = ax.imshow(v_nn, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=0, vmax=1200,
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict v')
    fig.colorbar(im, ax=ax, shrink=1)

    ################################
    ax = axs[2][0]
    im = ax.imshow(u_nn - ux, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('predict - obs u')
    fig.colorbar(im, ax=ax, shrink=1)


    ax = axs[2][1]
    im = ax.imshow(v_nn - uy, interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_title('predict - obs v')
    fig.colorbar(im, ax=ax, shrink=1)

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

    plt.show()
    #}}}
