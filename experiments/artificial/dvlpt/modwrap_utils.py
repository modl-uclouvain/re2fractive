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
from os import listdir
from os.path import isfile, join
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jobflow import job, Flow, Response


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



def feat_selec(
        X=None,
        Y=None,
        md_feat=None, 
        md_featselec_dir_path=Path('./moddata_featselec'),
        save=True,
        icycle=0, # index of active learning cycle
        ikf=0,    # index of the cross-validation fold
        **feature_selection_kwargs,
):

    # If it exists, let's load the MODData, otherw. let's define its path
    if not os.path.isdir(md_featselec_dir_path):
        os.makedirs(md_featselec_dir_path)
    md_featselec_path = (
        md_featselec_dir_path / f"md_featselec_{icycle:02}_{ikf:02}"
    )
    if md_featselec_path.exists():
        return MODData.load(md_featselec_path)

    if md_feat:
        md_featselec = MODData(df_featurized=md_feat.df_featurized)
        md_featselec.df_targets = md_feat.df_targets
    elif not X and not Y:
        md_featselec = MODData(df_featurized=X)
        md_featselec.df_targets = Y
    else:
        raise Exception("Need either a MODData or an input with a target dataframes!")

    md_featselec.feature_selection(**feature_selection_kwargs)

    if save:
        md_featselec.save("md_featselec_path")

    return md_featselec


def featurize(
        structures,
        ids,
        featurizer=None,
        featurizer_mode='single',
        n_jobs=2,
        df_feat_dir_path=Path('./featurized_df'),
        save=True,
        icycle=0, # index of active learning cycle
        ikf=0     # index of the cross-validation fold
):
    # If it exists, let's load the featurized df, otherw. let's define its path
    if not os.path.isdir(df_feat_dir_path):
        os.makedirs(df_feat_dir_path)
    df_feat_path = (
        df_feat_dir_path / f"df_feat_{icycle:02}_{ikf:02}"
    )
    if df_feat_path.exists():
        with open(df_feat_path, 'rb') as f:
            df_featurized = pickle.load(f)

    # If provided, use a Matminer Featurizer, otherw. use the default of MODData
    if featurizer:
        df_structures = pd.DataFrame.from_dict(
            {id: {'structure': s} for id, s in zip(ids, structures)}, orient='index'
        )

        df_featurized = featurizer.featurize(df_structures)
    else:
        
        md_feat = MODData(
            materials=structures,
            targets=[None]*len(structures),
            target_names=["Hoots"],
            structure_ids=ids,
        )
        md_feat.featurizer.featurizer_mode = featurizer_mode
        md_feat.featurize(n_jobs=n_jobs)
        df_featurized = md_feat
        
    
    if save:
        with open(df_feat_path, 'wb') as f:
            pickle.dump(df_featurized, f)

    return df_featurized



def load_feat_selec_df(md_feat_selec, selec=True, target=True, **load_kwargs):
    assert type(md_feat_selec)==str or type(md_feat_selec)==PosixPath or type(md_feat_selec)==MODData

    if type(md_feat_selec)==str or type(md_feat_selec)==PosixPath:
        md = MODData.load(md_feat_selec)
    else:
        md = deepcopy(md_feat_selec)
    
    # X = md.df_featurized
    test_size = 43 #VT TEST
    X = md.df_featurized.iloc[:test_size] #VT TEST

    if selec:
        # Y = md.df_targets
        Y = md.df_targets.iloc[:test_size] #VT TEST
        # Let's save the important information of the featurized MODData for easier reuse
        dct_md_info={
            'optimal_features':             md.optimal_features,
            'optimal_features_by_target':   md.optimal_features_by_target,
            'num_classes':                  md.num_classes
        }

        return X, Y, dct_md_info
    elif target and not selec:
        # Y = md.df_targets
        Y = md.df_targets.iloc[:test_size] #VT TEST
        return X, Y
    else:
        return X


def train_pool_random(X, Y, n=None, frac=0.1, state=None, axis=None):
    # Train dataset
    Xt = X.sample(n=n, frac=frac, random_state=state, axis=0)

    Yt = Y.filter(items=Xt.index.values, axis=0)

    # Pool dataset
    Xp = X.drop(Xt.index.values, axis=0)
    Yp = Y.drop(Yt.index.values, axis=0)

    return Xt, Yt, Xp, Yp

def bmk_scoring(
    bmk_results,
    md_feat_selec=None,
    Y=None,
    **load_kwargs
):

    if not Y:
        _, Y, _ = load_feat_selec_df(md_feat_selec, **load_kwargs)

    # Scores
    pred_mae  = []
    pred_rmse = []
    pred_spr  = []
    unc_mae   = []
    unc_rmse  = [] 

    for results in bmk_results:
        predictions, uncertainties = results
        if type(predictions)==dict:
            predictions = pd.DataFrame.from_dict(predictions)
            uncertainties = pd.DataFrame.from_dict(uncertainties)
        target_name = predictions.columns[0] # TODO: allow for multiple targets

        Ypred = Y.loc[predictions.index.values, :]

        # Scores
        pred_mae.append(mean_absolute_error(
            np.array(Ypred[target_name].values),
            np.array(predictions[target_name].values)
            )
            )
        pred_rmse.append(mean_squared_error(
            np.array(Ypred[target_name].values),
            np.array(predictions[target_name].values)
            )
            )
        pred_spr.append(spearmanr(
            np.array(Ypred[target_name].values),
            np.array(predictions[target_name].values)
            ).statistic
            )

        unc_mae.append(mean_absolute_error(
            np.zeros(np.array(uncertainties[target_name].values).shape),
            np.array(uncertainties[target_name].values)
            )
            )
        unc_rmse.append(mean_squared_error(
            np.zeros(np.array(uncertainties[target_name].values).shape),
            np.array(uncertainties[target_name].values)
            )
            )

    # Save scores
    scores = {
        'pred_mae': pred_mae,
        'pred_rmse': pred_rmse,
        'pred_spr': pred_spr,
        'unc_mae': unc_mae,
        'unc_rmse': unc_rmse,
    }

    return scores