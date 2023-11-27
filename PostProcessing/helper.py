import itertools
import pickle
import os
import re
import numpy as np
from utils import *
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd

def getListOfVaribles(x, keyword='weights', length=5): #{{{
    # find the index start with keyword
    index = [i for i, s in enumerate(x) if keyword in s]
    if len(index) == 1:
        # max possible length to lookup
        weights = x[index[0]:index[0]+length]
        # remove keyword
        weights[0] = weights[0].replace(keyword,'')
        # try to convert
        output = []
        for s in weights:
            if s:
                try:
                    output.append(float(s))
                except:
                    break

        return output
    else:
        return [] #}}}
def getFromRegex(x, expression): #{{{
    # get the NN architecture
    for s in x:
        nn = re.search(expression, s)
        if nn:
            try:
                return int(nn.group(0))
            except:
                return nn.group(0) #}}}
def upscale_by_weights(dataDict, keys, weights): #{{{
    for k,w in zip(keys,weights):
        dataDict[k] = (dataDict[k]/w) if k in dataDict else dataDict[k]
    return dataDict
    #}}}
def getDataDict(df, wh=5, lossWeightDict={}, dwfc=10, wf=[], prefix='SSA1D', NN='3NN', neurons=20, layers=4, projPath="./Models_Kubeflow/Models/", C_true=None): #{{{
    # weights
    wu = 5
    wC = 5
    # pick the intersetion of wf and the data in df
    if not wf:
        wf = np.sort(df[df['wh']==wh]['wf'].unique())
    else:
        wf = set(np.sort(df[df['wh']==wh]['wf'].unique()),).intersection(set(wf))

    weightsList = [(wu, wh, wC, w, w+dwfc) for w in wf]
    dataDict = {}

    # get the (key,weights) pair
    keys = [k for k in lossWeightDict.keys()]
    wids = [lossWeightDict[k] for k in keys]

    # compute normalization for C
    if C_true is None:
        Cnorm = 1
        NC = 1
    else:
        Cnorm = np.linalg.norm(C_true, 2)
        NC = len(C_true)

    # loop through experiments
    for weights in weightsList:
        # load all history data
        dataList = [load_history(projPath+name) for name in df[(df['wu']==weights[0]) &
                                                 (df['wf']==weights[3]) &
                                                 (df['wh']==weights[1]) &
                                                 (df['wfc-wf']==dwfc) &
                                                 (df['prefix']==prefix) &
                                                 (df['NN']==NN) &
                                                 (df['neurons']==neurons) &
                                                 (df['layers']==layers)
                                                ]['Name']]
        # compute the weights
        loss_weights = [10**(-weights[i]) if i>-1 else 1 for i in wids]
        # get the error in the final epoch
        final_epoch = [upscale_by_weights(get_final_errors(data), keys, loss_weights) for data in dataList]

        # adjuest the test resutls
        for e in final_epoch:
            e['test'] = (e['test']*Cnorm)**2/NC
#             e['test'] = e['test']
        # save data
        dataDict[weights] = {key: [i[key] for i in final_epoch] for key in final_epoch[0]}

    # colors indicates different weights combinations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lims = [[1e3, 1e8], [1e0, 1e6], [1e0, 1e6], [1e-1, 1e13], [1e4,1e7]]

    features = keys
    Nkey = len(keys)-1

    # plot
    fig, axs = plt.subplots(Nkey, Nkey, figsize=(20,16))

    for cid, weights in enumerate(weightsList):
        err=dataDict[weights]
        label = '$w_f=10^{-'+str(weights[3])+'}$'

        for i in range(Nkey):
            for j in range(i,Nkey):
                ax = axs[i][j]
                ax.scatter(err[features[j+1]], err[features[i]], c=colors[cid], label=label, s=20)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(lims[j+1])
                ax.set_ylim(lims[i])

        # add labels
        for i in range(len(features)-1):
            ax = axs[i][0]
            ax.set_ylabel(features[i])
        for j in range(len(features)-1):
            ax = axs[-1][j]
            ax.set_xlabel(features[j+1])

    ax.legend(bbox_to_anchor=(1.1, 1.5))

    return dataDict #}}}
def loadHistory(folder, projPath="./Models_Kubeflow/Models/", filename="history.json", N=None): #{{{
    '''
    load history data to a dictionary
    '''
    modelPath = projPath + folder + "/" + filename
    with open(modelPath,  'rb') as f:
        data = pickle.load(f)
    errorLength = getErrorLength(data)
    if N:
        return getLastNErrors(data,N), errorLength
    return data, errorLength#}}}
def getLastNErrors(data, N=10): #{{{
    keys = data.keys()
    return {k:data[k][-N-1:-1] for k in keys}
    #}}}
def getFinalErrors(data): #{{{
    return getLastNErrors(data, N=1)
    #}}}
def getErrorLength(data): #{{{
    keys = list(data.keys())
    return len(data[keys[0]])
    #}}}
def findAllExps(projPath="./Models_Kubeflow/Models/", dimensions = 1): #{{{
    # find all the experiments folders
    foldersList = os.listdir(projPath)
    df = pd.DataFrame(foldersList, columns=['Name'])

    # get info from the name
    df['sp'] = df['Name'].apply(lambda x: x.split('_'))
    # remove len(['sp'])>3
    splen = df['sp'].apply(lambda x: len(x))
    df = df[splen>3]

    # get all info from the folder name
    df['prefix'] = df['Name'].apply(lambda x: x.split('NN')[0][:-2])
    df['NN'] = df['sp'].apply(lambda x: getFromRegex(x, r"^[0-9]NN$"))
    df['layers'] = df['sp'].apply(lambda x: getFromRegex(x, r"^[0-9]+(?=x[0-9]+$)"))
    df['neurons'] = df['sp'].apply(lambda x: getFromRegex(x, r"(?<=[0-9]x|[00-99]x)[0-9]+"))
    df['Date'] = df['sp'].apply(lambda x: x[-2])
    df['Time'] = df['sp'].apply(lambda x: x[-1])
    df['weights'] = df['sp'].apply(getListOfVaribles)
    df['wu'] = df['weights'].apply(lambda x: x[0] if x else 0)
    df['wh'] = df['weights'].apply(lambda x: x[1] if x else 0)
    df['wC'] = df['weights'].apply(lambda x: x[2] if x else 0)
    df['wf'] = df['weights'].apply(lambda x: x[3] if x else 0)
    df['wfc'] = df['weights'].apply(lambda x: x[4] if x else 0)
    df['noise'] = df['sp'].apply(lambda x: getListOfVaribles(x,'noise',5))
    df = df.drop(columns=['sp'])
    df['wfc-wf'] = df['wfc']-df['wf']

    # remove name with 'seed'
    print('-> size before cleanup:', df.shape)
    df = df[df['Date'].str.strip()!='seed']
    print('-> size after cleanup:', df.shape)

    # convert date
    df['Date']=df['Date'].astype(int)
    df['Time']=df['Time'].astype(str)

    # different experiment types
    print('-> Keys before cleanup:', df['prefix'].unique())
    # remove unused data
    df = df[df['prefix'].str.contains("SSA2D|SSA1D")]
    print('-> Keys after cleanup:', df['prefix'].unique())

    # noise level, only use no noise cases
    print('-> Keys before cleanup: ', df['noise'].drop_duplicates())
    df = df[df['noise'].isin([[]])]
    print('-> Keys after cleanup:', df['noise'].drop_duplicates())

    # 1D or 2D problem
    if dimensions == 2:
        lossWeightDict = {'mse_u':0, 'mse_v':0, 'mse_H': 1, 'mse_s':1, 'mse_f1': 3, 'mse_f2': 3, 'mse_fc1':4, 'mse_fc2':4, 'test':-1}
        prefix = 'SSA2D'
    else:
        lossWeightDict = {'mse_u':0, 'mse_H': 1, 'mse_h':1, 'mse_f1': 3, 'mse_fc1':4, 'test':-1}
        prefix = 'SSA1D'

        # only look at SSA1D
        print('-> Keys before cleanup:', df['prefix'].unique())
        # remove unused data
        df = df[df['prefix'].str.strip() =="SSA1D"]
        print('-> Keys after cleanup:', df['prefix'].unique())

    # add columns for loading errors
    numCol = df.shape[1]
    df.insert(numCol, 'history length', 0)
    for k in lossWeightDict.keys():
        df.insert(numCol, k, 0.0)
        df.insert(numCol, k+"mean100", 0.0)
        df.insert(numCol, k+"100", 0.0)
        df[ k+"100"] = df[ k+"100"].astype(object)

    df.reset_index(inplace=True)
    return df #}}}
def upscaleByWeights(df, lossWeightDict={'mse_u':'wu', 'mse_H': 'wh', 'mse_h':'wh', 'mse_f1': 'wf', 'mse_fc1': 'wfc'}, N=100): #{{{
    for k in lossWeightDict:
        w = 10**(-df[lossWeightDict[k]])
        df[k] = df[k]/w
        newk = k + "mean" + str(N)
        df[newk] = df[newk]/w
    return df #}}}
def scaleMseC(df, C_true=None): #{{{
    # compute normalization for C
    if C_true is None:
        Cnorm = 1
        NC = 1
    else:
        Cnorm = np.linalg.norm(C_true, 2)
        NC = len(C_true)

    df['test'] = (df['test']*Cnorm)**2/NC
    df['testmean100'] = (df['testmean100']*Cnorm)**2/NC

    return df #}}}
def addErrors(df, N=100): #{{{
    '''
    load history.json for each rows in the df and add errors to the corresponding columns
    '''
    # loop over the dataframe
    for i in df.index:
        if i % 100 == 0:
            print(f"Loop over the whole dataframe: {i}/{df.shape[0]}")
        name = df.iloc[i]['Name']
        data, errL = loadHistory(name, N=N)
        df.at[i, 'history length'] = errL
        # save the last N data to df
        for k in data.keys():
            # mse in the last N errors
            newk = k + str(N)
            if newk in df:
                df.at[i, newk] = data[k][-N-1:-1]

    # get the last one and mean
    name = df.iloc[0]['Name']
    data, errL = loadHistory(name, N=N)
    for k in data.keys():
        refk = k + str(N)
        newk = k + "mean" + str(N)
        if (k in df) and (refk in df) and (newk in df):
            df[k] = df[refk].apply(lambda x: x[-1] if isinstance(x, list) else 0)
            df[newk] = df[refk].apply(lambda x: np.mean(x))

    # pre-processing
    df = upscaleByWeights(df, N=N)

    return df #}}}
