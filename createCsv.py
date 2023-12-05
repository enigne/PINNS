from PostProcessing import *
import pandas as pd

# all 1D results
df = findAllExps()
data = addErrors(df)
filename = "./Results/1Dresults.csv"
data.to_csv(filename, index_label=False)

# 2D flight track tests
modelFolder = 'Models_Kubeflow/2D/'
df = findAllExps(projPath=modelFolder, dimensions=2)
data = addErrors(df, projPath=modelFolder)
filename = "./Results/2Dresults.csv"
data.to_csv(filename, index_label=False)
