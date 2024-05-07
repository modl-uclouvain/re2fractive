def main():
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
    from importlib import reload

    import acquilib as acq

    reload(acq)

    from jobflow import job, Flow, run_locally
    from jobflow.managers.fireworks import flow_to_workflow
    from fireworks import LaunchPad


    from modwrap_utils import analysis, feat_selec, featurize, load_feat_selec_df, train_pool_random
    from modwrap_tasks import predict, select, train
    from modwrap_jf import train_predict, cycle, bmk_launch, new_cycle, campaign, bmk_analysis


    # 

    path_re2f = Path('/home/vtrinquet/Documents/Doctorat/JNB_Scripts_Clusters/NLO/HT/ref_idx/re2fractive')

    # Load the featurized MODData
    md_feat_selec_path = (path_re2f / 'humanguided' / 'v0' / 'mod.data_refeatselec_v0_v2')
    X, Y, dct_md_info = load_feat_selec_df(md_feat_selec_path)


    Xt, Yt, Xp, Yp = train_pool_random(X, Y, n=30, frac=None, state=42)

    model_params={
            'size_pop':2, # dflt 20
            'num_generations':2, # dflt 10
            'nested':0, # dflt = 5
            'n_jobs':2,
            'early_stopping':2, # dflt 4
            'refit':5, # dflt = 5
            'fast':False,
            }
    
    model_type=FitGenetic

    n_selec=5

    acquisition=acq.exploration
    # acquisition=None
    acq_kw = {"random_state": 42}

    # bmk_flow = bmk_analysis(
    #     bmk_results=None,
    #     X=None, Y=None, dct_md_info=None,
    #     md_feat_selec=md_feat_selec_path,
    #     train_index=Yt.index.values,
    #     model_type=FitGenetic, model_params=model_params,
    #     cv_k=2, cv_state=42, 
    #     bmk_dir_path='./benchmark',
    #     icycle=0,
    #     **acq_kw
    # )
    # run_locally(bmk_flow)

    campaign = campaign(
        md_feat_selec=md_feat_selec_path,
        train_index=Yt.index.values,
        model_type=model_type, model_params=model_params,
        n_selec=n_selec,
        acquisition=acquisition,        
        icycle=0,
        cv_k=2, cv_state=42, 
        **acq_kw
    )
    run_locally(campaign)

    # # convert the flow to a fireworks WorkFlow object
    # wf = flow_to_workflow(campaign)

    # # submit the workflow to the FireWorks launchpad (requires a valid connection)
    # lpad = LaunchPad.auto_load()
    # lpad.add_wf(wf)






if __name__ == "__main__":
    main()

# TODO: next step = clean things and put it into classes (Maker, ...) maybe allow for some abstraction to be able to use other ML models 