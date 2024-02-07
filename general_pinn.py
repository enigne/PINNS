from experiments import *

inputfile = "Helheim_Weertman_iT080_PINN_fastflow_CF"

# solve for u
outputfile = "SSA2D_solveU"
experiment_2D_3NN_test((8, 6, 8, 10, 18), epochADAM=1000000, seed=None, N_u=None, 
                        N_H=4000, N_C=4000, N_f=9000, NNeurons=20, NLayers=6, 
                        FlightTrack=False, inputFileName=inputfile, outputFileName=outputfile)

# solve for C
outputfile = "SSA2D_solveC"
experiment_2D_3NN_test((8, 6, 8, 10, 18), epochADAM=1000000, seed=None, N_u=4000, 
                        N_H=4000, N_C=None, N_f=9000, NNeurons=20, NLayers=6, 
                        FlightTrack=False, inputFileName=inputfile, outputFileName=outputfile)

# solve for H
outputfile = "SSA2D_solveH"
experiment_2D_3NN_test((8, 6, 8, 10, 18), epochADAM=1000000, seed=None, N_u=4000, 
                        N_H=None, N_C=4000, N_f=9000, NNeurons=20, NLayers=6, 
                        FlightTrack=False, inputFileName=inputfile, outputFileName=outputfile)
