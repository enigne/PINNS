from PostProcessing import *
import pandas as pd

df = findAllExps()
data = addErrors(df)
filename = "./Results/1Dresults.csv"
data.to_csv(filename, index_label=False)
