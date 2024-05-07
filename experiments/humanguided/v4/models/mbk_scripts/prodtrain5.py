def main():
    # python env: modnenv_v2
    
    import os
    import string
    import random
    import numpy as np
    from sklearn.model_selection import KFold
    from pathlib import Path
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

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    #n_jobs = int(0.7*int(os.environ["SLURM_CPUS_PER_TASK"]))
    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    # Trick of PP to prevent explosion of the threads
    def setup_threading():
    	import os
    	os.environ['OPENBLAS_NUM_THREADS'] = '1'
    	os.environ['MKL_NUM_THREADS'] = '1'
    	os.environ["OMP_NUM_THREADS"] = "1"
    	os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    	os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    setup_threading()
    
    cwd = os.getcwd()
    cur_v = 4

    modd_nac_new_path = (Path(cwd) / f'mod.data_refeatselec_v{cur_v}')
    md_naccarato_new = MODData.load(modd_nac_new_path)
    
    targets = ["refractive_index"]
    experiment_name = "GA_Rf0_Nstd5-" + "-".join(targets)
    scores = []

    
    models_dir_path = (Path(cwd) / "models")
    if not os.path.isdir(models_dir_path):
        os.mkdir(models_dir_path)
    fitgenetic_dir_path = (models_dir_path / "production")
    if not os.path.isdir(fitgenetic_dir_path):
        os.mkdir(fitgenetic_dir_path)
    
    model_path = (
        fitgenetic_dir_path / f"{experiment_name}_prod_v{cur_v}.pkl"
    )
    
    if model_path.exists():
        model = EnsembleMODNetModel.load(model_path)
    
    else:
        ga = FitGenetic(md_naccarato_new)
        model = ga.run(
            size_pop=20, # dflt
            num_generations=10, # dflt
            nested=5, # dflt = 5
            n_jobs=n_jobs,
            early_stopping=4, # dflt
            refit=0, # dflt = 5
            fast=False,
        )
        model.save(model_path)

if __name__ == "__main__":
    main()
