from experiments import *
inputlist = ["Helheim_Weertman_iT080_PINN_fastflow_CF_1500"] 
outputlist = ["SSA2D_FlightTrackH250"]
experiment_2D_3NN_test((8, 6, 8, 10, 18), epochADAM=500000, seed=None, N_u=4000, 
                       N_H=4000, N_C=None, N_f=9000, NNeurons=20, NLayers=6, 
                       FlightTrack=True, inputFileName=inputlist[i], outputFileName=outputlist[i])
