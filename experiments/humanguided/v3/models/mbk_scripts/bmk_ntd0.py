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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    modd_nac_new_path = (Path('.') / 'mod.data_refeatselec_v3')
    md_naccarato_new = MODData.load(modd_nac_new_path)
    
    kf = KFold(5, shuffle=True, random_state=42)
    
    targets = ["refractive_index"]
    experiment_name = "GA2_kfold_ensemble-" + "-".join(targets)
    scores = []
    
    
    cwd = "/globalscratch/ucl/modl/vtrinque/NLO/HT/ref_idx/re2fractive/humanguided/v3"
    models_dir_path = (Path(cwd) / "models")
    if not os.path.isdir(models_dir_path):
        os.mkdir(models_dir_path)
    fitgenetic_dir_path = (models_dir_path / "fitgenetic_2")
    if not os.path.isdir(fitgenetic_dir_path):
        os.mkdir(fitgenetic_dir_path)
    
    fig_bk_ntd0_path = (Path(cwd) / "models" / "fitgenetic_2" / f"{experiment_name}.png")
    if fig_bk_ntd0_path.exists():
        fig = Image(filename=fig_bk_ntd0_path)
        #display(fig)
    else:
        for ind, (train, test) in enumerate(
            kf.split(md_naccarato_new.df_featurized, y=md_naccarato_new.df_targets)
        ):
            train_moddata, test_moddata = md_naccarato_new.split((train, test))
            model_path = (
                Path(cwd) / "models" / "fitgenetic_2" / f"{experiment_name}_{ind}.pkl"
            )
            print(model_path)
            if model_path.exists():
                model = EnsembleMODNetModel.load(model_path)
    
            else:
                ga = FitGenetic(train_moddata)
                model = ga.run(
                    size_pop=20, # dflt
                    num_generations=10, # dflt
                    nested=0, # dflt = 5
                    n_jobs=2,
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
        # plt.xlim((-1,30))
        # plt.ylim((-1,30))
        plt.title(f"MAE: {np.mean(scores):.3f}±{np.std(scores):.3f}")
    
        plt.savefig(fig_bk_ntd0_path)
        plt.savefig(Path(str(fig_bk_ntd0_path).replace("png", "pdf")))


if __name__ == "__main__":
    main()
