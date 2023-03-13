import itertools
import pickle
from os import listdir

def load_history(projPath, filename="history.json"): #{{{
    '''
    load history data
    '''
    modelPath = projPath + "/" + filename
    with open(modelPath,  'rb') as f:
        data = pickle.load(f)
    return data#}}}
def load_history_at_weights(weights, projPath="./Models/", prefix="SSA1D_weights", filename="history.json"): #{{{
    '''
    load all history data with given weights 
    '''
    # get all the folders in the projPath
    foldersList = listdir(projPath)

    # get filename pattern
    filename_pattern = prefix + "".join([str(w)+"_" for w in weights])

    # filter
    filteredList = [f for f in foldersList if filename_pattern in f]

    return [load_history(projPath+f) for f in filteredList] #}}}
def get_final_errors(data):
    keys = data.keys()
    return {k:data[k][-1] for k in keys}



if __name__ == "__main__":
    # weights: u, h/H, C, f1, fc1 
    wu = [5]
    wh = [3]
    wC = [5]
    wf1 = [8]
    wfc1 = [14]
    dataList = []
    errorDict = {}
    
    weightsList = list(itertools.product(*[wu,wh,wC,wf1,wfc1]))
    
    # loop through experiments 
    for weights in weightsList:
        # loss_weights = [10**(-w) for w in weights]
        dataList = load_history_at_weights(weights)
        errors = errorDict.setdefault(weights, [])
        errors.append([get_final_errors(data) for data in dataList])
