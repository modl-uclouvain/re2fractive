# python env: modnenv_v2

import os
import string
import random
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import json 
import pandas as pd
from copy import deepcopy
from modnet.models import EnsembleMODNetModel
from modnet.preprocessing import MODData
from modnet.hyper_opt import FitGenetic
from monty.serialization import dumpfn, loadfn
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from scipy.stats import spearmanr
from IPython.display import Image
from tqdm import tqdm
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from os import listdir
from os.path import isfile, join

def analysis(
      scores_dir_path=Path('./benchmark/scores')
      ):

    if (scores_dir_path / f"scores_overall.png").exists() and\
       (scores_dir_path / f"scores_overall.pdf").exists() and\
       (scores_dir_path / f"scores_unc_overall.png").exists() and\
       (scores_dir_path / f"scores_unc_overall.pdf").exists():
       print('Already analyzed!')
       return


    scores_all = {
       'mae_folds': [],
       'rmse_folds': [],
       'spr_folds': [],
       'mae_unc_folds': [],
       'rmse_unc_folds': [],
       'mae_avg': [],
       'rmse_avg': [],
       'spr_avg': [],
       'mae_unc_avg': [],
       'rmse_unc_avg': [],
    }

    scoresfiles = [f for f in listdir(scores_dir_path) if isfile(join(scores_dir_path, f)) and any(ch.isdigit() for ch in f)]
    for f in scoresfiles:
        scores_path = (scores_dir_path / f)

        with open(scores_path) as f:
            scores = json.load(f)

        scores_all['mae_folds'].append(scores['pred_mae'])
        scores_all['rmse_folds'].append(scores['pred_rmse'])
        scores_all['spr_folds'].append(scores['pred_spr'])
        scores_all['mae_unc_folds'].append(scores['unc_mae'])
        scores_all['rmse_unc_folds'].append(scores['unc_rmse'])
        scores_all['mae_avg'].append(np.mean(scores['pred_mae']))
        scores_all['rmse_avg'].append(np.mean(scores['pred_rmse']))
        scores_all['spr_avg'].append(np.mean(scores['pred_spr']))
        scores_all['mae_unc_avg'].append(np.mean(scores['unc_mae']))
        scores_all['rmse_unc_avg'].append(np.mean(scores['unc_rmse']))
    
    x = range(len(scores_all['mae_avg']))

    # Create subplots and unpack the Axes object
    fig, ax = plt.subplots()

    # Plot the data using the ax object
    ax.plot(x, -np.array(scores_all['mae_avg']), label='-MAE')
    ax.plot(x, -np.array(scores_all['rmse_avg']), label='-RMSE')
    ax.plot(x, scores_all['spr_avg'], label='+SPR')

    # Set labels for the axes
    ax.set_xlabel('# AL cycles', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)

    # Add a legend
    ax.legend(fontsize=12)
    fig.savefig((scores_dir_path / f"scores_overall.png"))
    fig.savefig((scores_dir_path / f"scores_overall.pdf"))

    # Create subplots and unpack the Axes object
    fig, ax = plt.subplots()

    # Plot the data using the ax object
    ax.plot(x, -np.array(scores_all['mae_unc_avg']), label='-MAE')
    ax.plot(x, -np.array(scores_all['rmse_unc_avg']), label='-RMSE')

    # Set labels for the axes
    ax.set_xlabel('# AL cycles', fontsize=14)
    ax.set_ylabel('Score uncertainty', fontsize=14)

    # Add a legend
    ax.legend(fontsize=12)
    fig.savefig((scores_dir_path / f"scores_unc_overall.png"))
    fig.savefig((scores_dir_path / f"scores_unc_overall.pdf"))

    return scores_all
    
if __name__ == "__main__":
    _ = analysis()

