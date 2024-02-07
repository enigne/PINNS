from experiments import *

import itertools
# weights: u, h/H, C, f1, fc1
wu = [8]
wh = [6]
wC = [8]
wf1 = list(range(4,18,1))
weights = [(wu[0],wh[0],wC[0], w, w+10) for w in wf1]

for i in range(25):
    for w in weights:
        experiment_1D_3NN_hyperparameter_search(list(w), epochADAM=100000, N_u=100, N_f=500, 
                                                seed=None, log_frequency=10000, NLayers=6, NNeurons=20)
