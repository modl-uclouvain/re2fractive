def main():
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
    from monty.serialization import dumpfn, loadfn, MontyDecoder
    from optimade.adapters import Structure as optim_Structure
    from pymatgen.ext.matproj import MPRester
    from pymatgen.core.structure import Structure
    from pymongo import MongoClient
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
    
    client = MongoClient('emma.pcpm.ucl.ac.be', 27017, username='structDB', password='structDB2018', authSource='structure_databases', authMechanism='SCRAM-SHA-1')
    db = client.structure_databases
    
    
    collection = db.full_unique
    
    collections_chemenv = {"csd": db.csd_chemenv,
                           "cod": db.cod_chemenv,
                           "icsd": db.icsd_chemenv,
                           "pauling": db.pauling_chemenv}
    
    collections_unique = {"csd": db.csd_unique,
                           "cod": db.cod_unique,
                           "icsd": db.icsd_unique,
                           "pauling": db.pauling_unique}
    
    collections_raw = {"csd": db.csd_unique,
                       "cod": db.cod_unique,
                       "icsd": db.icsd_unique,
                       "pauling": db.pauling_unique}
    
    for mark in np.arange(0, 1166730, 25000):
        structures_mshp = []
        path_lst_mshp_struc = Path(f'{mark:07}_lst_mshp_struc.pkl')
    
        if path_lst_mshp_struc.exists():
            # with open(path_lst_mshp_struc, 'rb') as f:
            #     structures_mshp = pickle.load(f)
            pass
        else:
            for i, r in tqdm(enumerate(collection.find())):
                if i>=mark and i<mark+25000:
                    data = collections_unique[r["source"]].find_one({"unique_id": r["source_id"]})
        
                    structures_mshp.append(Structure.from_dict(data["structure"]))
                elif i>=mark+25000:
                    break
                else:
                    pass
            with open(path_lst_mshp_struc, 'wb') as f:
                pickle.dump(structures_mshp, f)

if __name__=="__main__":
    main()
