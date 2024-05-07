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
    
    cur_v = 4
    cwd = os.getcwd()

    modd_nac_new_path = (Path(cwd) / f'mod.data_refeatselec_v{cur_v}')
    md_naccarato_new = MODData.load(modd_nac_new_path)
    
    kf = KFold(5, shuffle=True, random_state=42)
    
    targets = ["refractive_index"]
    experiment_name = "GA3_kfold_ensemble-" + "-".join(targets)
    scores = []
    
    
    models_dir_path = (Path(cwd) / "models")
    if not os.path.isdir(models_dir_path):
        os.mkdir(models_dir_path)
    fitgenetic_dir_path = (models_dir_path / "fitgenetic_3")
    if not os.path.isdir(fitgenetic_dir_path):
        os.mkdir(fitgenetic_dir_path)
    
    fig_bk_ntd5_path = (fitgenetic_dir_path / f"{experiment_name}.png")
    if fig_bk_ntd5_path.exists():
        fig = Image(filename=fig_bk_ntd5_path)
        #display(fig)
    else:
        for ind, (train, test) in enumerate(
            kf.split(md_naccarato_new.df_featurized, y=md_naccarato_new.df_targets)
        ):
            train_moddata, test_moddata = md_naccarato_new.split((train, test))
            model_path = (
                fitgenetic_dir_path / f"{experiment_name}_{ind}.pkl"
            )
            print(model_path)
            if model_path.exists():
                model = EnsembleMODNetModel.load(model_path)
    
            else:
                ga = FitGenetic(train_moddata)
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
    
            scores.append(model.evaluate(test_moddata))
    
            predictions, uncertainties = model.predict(test_moddata, return_unc=True)
    
            plt.scatter(
                test_moddata.df_targets.values.ravel(),
                predictions.values.ravel(),
            )
            plt.errorbar(
                test_moddata.df_targets.values.ravel(),
                predictions.values.ravel(),
                yerr=uncertainties.values.ravel(),
                ls="none",
            )
    
        print("="*10 + f" {experiment_name} " + "="*10)
        print("Training complete.")
        print("Training complete.")
        print(f"Accuracy: {np.mean(scores):.3f}±{np.std(scores):.3f}")
    
        plt.plot(
            np.linspace(
                np.min(md_naccarato_new.df_targets.values),
                np.max(md_naccarato_new.df_targets.values),
                3,
            ),
            np.linspace(
                np.min(md_naccarato_new.df_targets.values),
                np.max(md_naccarato_new.df_targets.values),
                3,
            ),
            color="black",
            ls="--",
        )
    
        plt.ylabel("Predicted $n$")
        plt.xlabel("Computed $n$")
        plt.title(f"MAE: {np.mean(scores):.3f}±{np.std(scores):.3f}")
    
        plt.savefig(fig_bk_ntd5_path)
        plt.savefig(Path(str(fig_bk_ntd5_path).replace("png", "pdf")))

if __name__ == "__main__":
    main()
