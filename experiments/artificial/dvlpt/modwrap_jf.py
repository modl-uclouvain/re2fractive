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

from modwrap_utils import analysis, feat_selec, featurize, load_feat_selec_df, train_pool_random, bmk_scoring
from modwrap_tasks import predict, select, train



def train_predict(
    md_feat_selec,
    train_index,
    model_type, model_params,
    model_dir_path,
    results_dir_path,
    pred_index=None,
    icycle=0,
    icv=None,
):
    train_job = train(
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        model_type=model_type,
        model_params=model_params,
        model_dir_path=model_dir_path,
        icycle=icycle,
        icv=icv
    )

    predict_job = predict(
        model_path=train_job.output,
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        pred_index=pred_index,
        results_dir_path=results_dir_path,
        icycle=icycle,
        icv=icv
    )

    return Flow([train_job, predict_job], output=predict_job.output)


def cycle(
    md_feat_selec,
    train_index,
    model_type, model_params,
    n_selec,
    acquisition,        
    icycle=0,
    working_dir='.',
    **acquisition_load_kwargs
):

    model_dir_path=Path(working_dir + '/production/models')
    results_dir_path=Path(working_dir + '/production/results')
    working_dir = os.getcwd()
    train_predict_flow =  train_predict(
        md_feat_selec,
        train_index,
        model_type, model_params,
        model_dir_path=model_dir_path,
        results_dir_path=results_dir_path,
        icycle=icycle
    )

    select_job = select(
            results=train_predict_flow.output,
            train_index=train_index,
            md_feat_selec=md_feat_selec,
            n=n_selec,
            frac=None,
            acquisition=acquisition,
            **acquisition_load_kwargs
            )
    

    return Flow([train_predict_flow, select_job], output=(select_job.output))
    



def bmk_launch(
    X=None, Y=None, dct_md_info=None,
    md_feat_selec=None,
    train_index=None,
    model_type=FitGenetic, model_params=None,
    cv_k=5, cv_state=None, 
    bmk_dir_path='./benchmark',
    icycle=0,
    **load_kwargs
):
    if not X and not Y and not dct_md_info:
        Xall, Yall, dct_md_info = load_feat_selec_df(md_feat_selec=md_feat_selec, **load_kwargs)
        X = Xall.filter(items=train_index, axis=0)
        Y = Yall.filter(items=train_index, axis=0)

    assert type(bmk_dir_path)==str or type(bmk_dir_path)==PosixPath
    bmk_dir_path = Path(bmk_dir_path)


    if not os.path.isdir(bmk_dir_path):
        os.mkdir(bmk_dir_path)
    models_dir_path = (bmk_dir_path / 'models')
    if not os.path.isdir(models_dir_path):
        os.mkdir(models_dir_path)
    scores_dir_path = (bmk_dir_path / 'scores')
    if not os.path.isdir(scores_dir_path):
        os.mkdir(scores_dir_path)
    scores_path = (scores_dir_path / f'scores_{icycle:02}.json')
    results_dir_path = (bmk_dir_path / 'results')
    if not os.path.isdir(results_dir_path):
        os.mkdir(results_dir_path)



    bmk_jobs = []
    bmk_jobs_out = []
    kf = KFold(cv_k, shuffle=True, random_state=cv_state)
    for icv, (train, test) in enumerate(
        kf.split(X=X, y=Y)
    ):

        train_idx = list(Y.iloc[train].index.values)
        test_idx = list(Y.iloc[test].index.values)
        
        results_path = (
            results_dir_path / f"results_{icycle:02}_{icv:02}.pkl"
        )

        if results_path.exists():
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            bmk_jobs_out.append(results)
            continue

        model_path = (
            models_dir_path / f"model_{icycle:02}_{icv:02}.pkl"
        )
        if model_path.exists():
            t_p_flow = predict(
                model_path=model_path,
                md_feat_selec=md_feat_selec,
                train_index=None,
                pred_index=test_idx,
                results_dir_path=results_dir_path,
                icycle=icycle,
                icv=icv
            )
        else:
            t_p_flow = train_predict(
                md_feat_selec=md_feat_selec,
                train_index=train_idx,
                model_type=model_type, model_params=model_params,
                model_dir_path=models_dir_path,
                results_dir_path=results_dir_path,
                pred_index=test_idx,
                icycle=icycle,
                icv=icv,
            )

        bmk_jobs.append(t_p_flow)
        bmk_jobs_out.append(t_p_flow.output)
        

    bmk_run_flow = Flow(bmk_jobs, output=bmk_jobs_out)

    bmk_analysis_job = bmk_analysis(
        bmk_results=bmk_run_flow.output,
        X=X, Y=Y, dct_md_info=dct_md_info,
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        model_type=model_type, model_params=model_params,
        cv_k=cv_k, cv_state=cv_state, 
        bmk_dir_path=bmk_dir_path,
        icycle=icycle,
        **load_kwargs
    )

    return Flow([bmk_run_flow, bmk_analysis_job], output=bmk_analysis_job.output)


@job
def bmk_analysis(
    bmk_results=None,
    X=None, Y=None, dct_md_info=None,
    md_feat_selec=None,
    train_index=None,
    model_type=FitGenetic, model_params=None,
    cv_k=5, cv_state=None, 
    bmk_dir_path='./benchmark',
    icycle=0,
    **load_kwargs
):

    assert type(bmk_dir_path)==str or type(bmk_dir_path)==PosixPath
    bmk_dir_path = Path(bmk_dir_path)
    if not os.path.isdir(bmk_dir_path):
        os.mkdir(bmk_dir_path)

    scores_dir_path = (bmk_dir_path / 'scores')
    if not os.path.isdir(scores_dir_path):
        os.mkdir(scores_dir_path)
    scores_path = (scores_dir_path / f'scores_{icycle:02}.json')

    results_dir_path = (bmk_dir_path / 'results')
    if not os.path.isdir(results_dir_path):
        os.mkdir(results_dir_path)

    if scores_path.exists():
        with open(scores_path) as f:
          scores = json.load(f)
        return scores

    if not X and not Y and not dct_md_info:
        Xall, Yall, dct_md_info = load_feat_selec_df(md_feat_selec=md_feat_selec, **load_kwargs)
        X = Xall.filter(items=train_index, axis=0)
        Y = Yall.filter(items=train_index, axis=0)
    
        
    results_files = [(results_dir_path / f"results_{icycle:02}_{icv:02}.pkl") for icv in range(cv_k)]
    if not all(list(map(os.path.isfile,results_files))) and not bmk_results:
        return Response(replace=bmk_launch(
            X=None, Y=None, dct_md_info=None,
            md_feat_selec=md_feat_selec,
            train_index=train_index,
            model_type=model_type, model_params=model_params,
            cv_k=cv_k, cv_state=cv_state, 
            bmk_dir_path=bmk_dir_path,
            icycle=icycle,
            **load_kwargs
        ))
    
    if all(list(map(os.path.isfile,results_files))) and not bmk_results:
        bmk_results = []
        for results_file in results_files:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            bmk_results.append(results)
    
    scores = bmk_scoring(
        bmk_results=bmk_results,
        md_feat_selec=md_feat_selec,
        Y=None,
        **load_kwargs
    )

    with open(scores_path, 'w') as f:
      json.dump(scores, f)

    return scores
    


@job
def new_cycle(
    md_feat_selec,
    train_index,
    train_index_new,
    model_type, model_params,
    n_selec,
    acquisition,        
    icycle,
    cv_k, cv_state,
    working_dir,
    **acquisition_load_kwargs
):
    # The training set already contains the whole pool dataset
    if not train_index_new:
        return "DONE"
    else:
        return Response(addition=campaign(
            md_feat_selec=md_feat_selec,
            train_index=train_index_new,
            model_type=model_type, model_params=model_params,
            n_selec=n_selec,
            acquisition=acquisition,        
            icycle=icycle+1,
            cv_k=cv_k, cv_state=cv_state,
            working_dir=working_dir,
            **acquisition_load_kwargs
        ))


# def campaign(
#     md_feat_selec,
#     train_index,
#     model_type, model_params,
#     n_selec,
#     acquisition,        
#     icycle=0,
#     **acquisition_load_kwargs
# ):
#     cycle_flow = cycle(
#         md_feat_selec=md_feat_selec,
#         train_index=train_index,
#         model_type=model_type, model_params=model_params,
#         n_selec=n_selec,
#         acquisition=acquisition,        
#         icycle=icycle,
#         **acquisition_load_kwargs
#     )

#     new_cycle_job = new_cycle(
#         md_feat_selec=md_feat_selec,
#         train_index=train_index,
#         train_index_new=cycle_flow.output,
#         model_type=model_type, model_params=model_params,
    #     n_selec=n_selec,
    #     acquisition=acquisition,        
    #     icycle=icycle,
    #     **acquisition_load_kwargs
    # )

    # return Flow([cycle_flow, new_cycle_job])

def campaign(
    md_feat_selec,
    train_index,
    model_type, model_params,
    n_selec,
    acquisition,        
    icycle=0,
    cv_k=5, cv_state=42,
    working_dir=None,
    **acquisition_load_kwargs
):

    if not working_dir:
        working_dir = os.getcwd()

    bmk_flow = bmk_analysis(
        bmk_results=None,
        X=None, Y=None, dct_md_info=None,
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        model_type=model_type, model_params=model_params,
        cv_k=cv_k, cv_state=cv_state, 
        bmk_dir_path=Path(working_dir + '/benchmark'),
        icycle=icycle,
        **acquisition_load_kwargs
    )

    cycle_flow = cycle(
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        model_type=model_type, model_params=model_params,
        n_selec=n_selec,
        acquisition=acquisition,        
        icycle=icycle,
        working_dir=working_dir,
        **acquisition_load_kwargs
    )

    new_cycle_job = new_cycle(
        md_feat_selec=md_feat_selec,
        train_index=train_index,
        train_index_new=cycle_flow.output,
        model_type=model_type, model_params=model_params,
        n_selec=n_selec,
        acquisition=acquisition,        
        icycle=icycle,
        cv_k=cv_k, cv_state=cv_state,
        working_dir=working_dir,
        **acquisition_load_kwargs
    )

    return Flow([bmk_flow, cycle_flow, new_cycle_job])

