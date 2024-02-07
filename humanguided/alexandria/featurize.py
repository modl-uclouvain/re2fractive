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

def main():
    path_cwd = Path(os.getcwd())
    path_df_agm = (path_cwd / 'df_agm_jan24.pkl')

    if path_df_agm.exists():
        with open(path_df_agm, 'rb') as f:
            df_agm = pickle.load(f)

    agm_structures = []
    for agmid, mat in df_agm.iterrows():
        s = Structure.from_dict(mat['structure'])
        agm_structures.append(s)
    
    agm_ids = df_agm.index.values.tolist()

    _ = featurize(
            structures=agm_structures,
            ids=agm_ids,
            featurizer=None,
            featurizer_mode='single',
            n_jobs=,
            df_feat_dir_path=(path_cwd / 'feat_mdd_single'),
            save=True,
            icycle=0, # index of active learning cycle
            ikf=0     # index of the cross-validation fold
    )
    


if __name__=="__main__":
    main()
