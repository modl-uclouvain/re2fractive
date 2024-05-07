
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

from modwrap_utils import analysis, feat_selec, featurize, load_feat_selec_df, train_pool_random


@job
def predict(
        model_path,
        X=None,
        md_feat_selec=None,
        train_index=None,
        pred_index=None,
        results_dir_path = Path('./production/results'),
        icycle=0,
        icv=0,
        **load_kwargs
):


    if not X:
        Xall, _, _ = load_feat_selec_df(md_feat_selec=md_feat_selec, **load_kwargs)
        if pred_index:
            X = Xall.filter(items=pred_index, axis=0)
        else:
            X = Xall.drop(labels=train_index, axis=0)

    

    if X.shape[0]==0:
        return None


    model = EnsembleMODNetModel.load(model_path)

    # New MODData to predict
    md_p = MODData(
        materials       = [None]*X.shape[0],
        df_featurized   = X,
        structure_ids   = X.index.values.flatten(),
    )

    results_dir_path = Path(results_dir_path)
    if not os.path.isdir(results_dir_path):
        os.makedirs(results_dir_path)

    if icv!=None:
        results_path = (
            results_dir_path / f"results_{icycle:02}_{icv:02}.pkl"
        )
    else:
        results_path = (
            results_dir_path / f"results_{icycle:02}.pkl"
        )
    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        # Predict the filtered MP
        results = model.predict(md_p, return_unc=True)

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    return results



# TODO: save the selected instances (at least their ids) with their scores, either as output of the Flow or as pkl file in a new folder
@job
def select(results, 
           Xt=None, Xp=None,
           X=None, dct_md_info=None,
           train_index=None,
           md_feat_selec=None,
           acquisition=None, frac=0.05, n=None, 
           **acquisition_load_kwargs):

    if not results:
        return None

    if not Xt and not Xp and train_index:
        if md_feat_selec:
            X, _, _ = load_feat_selec_df(md_feat_selec=md_feat_selec, **acquisition_load_kwargs)
        Xt = X.filter(items=train_index, axis=0)
        Xp = X.drop(labels=train_index, axis=0)


    # unpacking the results
    predictions, uncertainties = results
    predictions = pd.DataFrame.from_dict(predictions)
    uncertainties = pd.DataFrame.from_dict(uncertainties)

    if not acquisition:
        if frac:
            n=int(len(Xt)*frac)
        if n>predictions.shape[0]:
            n = predictions.shape[0]
        id_selected = predictions.sample(n=n, axis=0, **acquisition_load_kwargs).index.values
    else:
        scored = acquisition(predictions, uncertainties, **acquisition_load_kwargs)
        scored = scored.sort_values(by='score',ascending=False).dropna()
        if n:
            id_selected = scored.index.values[:n]
        else:
            id_selected = scored.index.values[:int(len(Xt)*frac)]
        
    
    Xs = Xp.filter(items=id_selected, axis=0)

    return pd.concat([Xt, Xs], axis=0).index.values




@job
def train(
    X=None, Y=None, dct_md_info=None,
    md_feat_selec=None,
    train_index = None,
    model_type=FitGenetic, model_params=None, model_dir_path = (Path(".") / "production" / "models"),
    icycle=0,
    icv=0,
    **load_kwargs
    ):

    if not X and not Y and not dct_md_info:
        Xall, Yall, dct_md_info = load_feat_selec_df(md_feat_selec=md_feat_selec, **load_kwargs)
        X = Xall.filter(items=train_index, axis=0)
        Y = Yall.filter(items=train_index, axis=0)


    # Create a new training MODData
    md_t = MODData(df_featurized=X)
    md_t.df_targets                 = Y
    md_t.optimal_features           = dct_md_info['optimal_features']
    md_t.optimal_features_by_target = dct_md_info['optimal_features_by_target']
    md_t.num_classes                = dct_md_info['num_classes']

    # Load or train and save
    model_dir_path = Path(model_dir_path)
    if not os.path.isdir(model_dir_path):
        os.makedirs(model_dir_path)
    if icv!=None:
        model_path = (
            model_dir_path / f"model_{icycle:02}_{icv:02}.pkl"
        )
    else:
        model_path = (
            model_dir_path / f"model_{icycle:02}.pkl"
        )


    if model_path.exists():
        print("Model already exists!")
        model = EnsembleMODNetModel.load(model_path)
    else:
        ga = model_type(md_t)
        model = ga.run(
            **model_params
        )
        model.save(model_path)

    return model_path

