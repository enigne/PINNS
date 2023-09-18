from experiments import *
import itertools
# weights: u, h/H, C, f1, fc1 
wu = [5]
wh = [3]
wC = [5]
wf1 = list(range(6,8,2))
# seed = 1234 # not set random seed
weights = [(wu[0],wh[0],wC[0], w, w+6) for w in wf1] 
weights

for i in range(1):
    for w in weights:
        experiment_2D_3NN_hyperparameter_search(list(w), epochADAM=200000, epochLBFGS=100000, N_u=2000, N_f=4000, seed=None, log_frequency=10000, NLayers=8)

