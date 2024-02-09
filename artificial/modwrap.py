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




def actilearn(
    structures,
    ids,
    X,
    Y,
    md_feat,
    md_featselec,
    start_frac,
    start_n,
    start_set,          #TODO
    start_state,
    ncycles,
    accuracy,
    accuracy_type,
    end_set,
    model_type,
    model_params,
    cv_k,
    cv_state,
    acquisition,
    acquisition_kwargs,
    acquisition_n,
    acquisition_frac,
    featurize_cycle,    #TODO
    featurize_cv,       #TODO
    featselec_cycle,    #TODO
    featselec_cv,       #TODO
):
    
    if [structures, X, md_feat, md_featselec].count(None) != 3:
        raise TypeError('Only one of either structures, X, md_feat, or md_featselec must be provided.')

    dct_md_info=None
    if md_featselec:
        X, Y, dct_md_info = load_feat_selec_df(md_featselec, selec=True, target=True)
    elif md_feat and not Y:
        X, Y = load_feat_selec_df(md_feat, selec=False, target=True)
    elif md_feat and Y:
        X = load_feat_selec_df(md_feat, selec=False, target=False)
    elif structures and Y:
        X = featurize(structures=structures,
                      ids=ids)
    else:
        raise Exception('You should either provide a feature selected MODData, '
                        'or a featurized MODData with/without targets, '
                        'or a list of structures with the corresponding ids and targets. '
                        'This was not the case.')


    # Choose a subset of the data to start the learning iterations
    Xt, Yt, Xp, Yp = train_pool_random(X, Y, n=start_n, frac=start_frac, state=start_state)
    # TODO: start_set

    for icycle in range(ncycles):
        # Benchmark 
        scores_bk = benchmk(
            X=Xt, Y=Yt, dct_md_info=dct_md_info,
            model_type=model_type,
            model_params=model_params,
            cv_k=cv_k, cv_state=cv_state, 
            # bmk_dir_path=
            icycle=icycle
            )

         # Train
        model = train(
            X=Xt, Y=Yt, dct_md_info=dct_md_info,
            model_type=model_type,
            model_params=model_params,
            # model_dir_path=".", 
            icycle=icycle
            )

        # End conditions
        if Xp.shape[0] == 0:
            break

        # Predict the pool data
        results = predict(
            X=Xp, model=model,
            #    results_dir_path=
            icycle=icycle
            )

        # End conditions
        if end_set:
            include_end_set = set(end_set).issubset(set(Xt.index.values))
        else:
            include_end_set = False
        if accuracy_type:
            reached_accuracy = max(scores_bk[accuracy_type])<=accuracy
        else:
            reached_accuracy = False
        if ncycles:
            reached_ncycles = (icycle==ncycles-1)
        else:
            reached_ncycles = False
        if      reached_accuracy \
            or  reached_ncycles \
            or  include_end_set \
        :
            break
        
        # Score, rank, and select the new data to be added to training
        Xt, Yt, Xp, Yp = select(
            Xt=Xt,
            Yt=Yt,
            Xp=Xp,
            Yp=Yp,
            n=acquisition_n,
            frac=acquisition_frac,
            results=results,
            acquisition=acquisition,
            acquisition_kwargs=acquisition_kwargs
            )
        
    
    return Xt, Yt, Xp, Yp, results, model, scores_bk  




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


def benchmk(
        X, Y, dct_md_info,
        model_type=FitGenetic, model_params=None,
        cv_k=5, cv_state=None, 
        bmk_dir_path='./benchmark',
        icycle=0
        ):

    assert type(bmk_dir_path)==str or type(bmk_dir_path)==PosixPath
    bmk_dir_path = Path(bmk_dir_path)


    if not os.path.isdir(bmk_dir_path):
        os.mkdir(bmk_dir_path)
    figs_dir_path = (bmk_dir_path / 'figs')
    if not os.path.isdir(figs_dir_path):
        os.mkdir(figs_dir_path)
    fig_path = (figs_dir_path / f'fig_{icycle:02}.png')
    models_dir_path = (bmk_dir_path / 'models')
    if not os.path.isdir(models_dir_path):
        os.mkdir(models_dir_path)
    scores_dir_path = (bmk_dir_path / 'scores')
    if not os.path.isdir(scores_dir_path):
        os.mkdir(scores_dir_path)
    scores_path = (scores_dir_path / f'scores_{icycle:02}.json')


    if fig_path.exists():
        fig = Image(filename=fig_path)
        #display(fig)
        with open(scores_path) as f:
          scores = json.load(f)
        return scores
    else:

        # Create a new training MODData that will be split into k folds
        md_t = MODData(df_featurized=X)
        md_t.df_targets                 = Y
        md_t.optimal_features           = dct_md_info['optimal_features']
        md_t.optimal_features_by_target = dct_md_info['optimal_features_by_target']
        md_t.num_classes                = dct_md_info['num_classes']

        # Scores
        pred_mae  = []
        pred_rmse = []
        pred_spr  = []
        unc_mae   = []
        unc_rmse  = []

        # Figure
        plt.figure(figsize=(10,5))

        kf = KFold(cv_k, shuffle=True, random_state=cv_state)
        for ind, (train, test) in enumerate(
            kf.split(X=X, y=Y)
        ):
            # KFold
            train_moddata, test_moddata = md_t.split((train, test))


            model_path = (
                models_dir_path / f"model_{icycle:02}_{ind:02}.pkl"
            )
            if model_path.exists():
                model = EnsembleMODNetModel.load(model_path)
            else:
                ga = model_type(train_moddata)
                model = ga.run(
                    **model_params
                )
                model.save(model_path)

            predictions, uncertainties = model.predict(test_moddata, return_unc=True)

            # Scores
            pred_mae.append(mean_absolute_error(
                np.array(test_moddata.df_targets[test_moddata.target_names[0]].values),
                np.array(predictions[test_moddata.target_names[0]])
                )
                )
            pred_rmse.append(mean_squared_error(
                np.array(test_moddata.df_targets[test_moddata.target_names[0]].values),
                np.array(predictions[test_moddata.target_names[0]])
                )
                )
            pred_spr.append(spearmanr(
                np.array(test_moddata.df_targets[test_moddata.target_names[0]].values),
                np.array(predictions[test_moddata.target_names[0]])
                ).statistic
                )

            unc_mae.append(mean_absolute_error(
                np.zeros(np.array(uncertainties[test_moddata.target_names[0]]).shape),
                np.array(uncertainties[test_moddata.target_names[0]])
                )
                )
            unc_rmse.append(mean_squared_error(
                np.zeros(np.array(uncertainties[test_moddata.target_names[0]]).shape),
                np.array(uncertainties[test_moddata.target_names[0]])
                )
                )
            

            # Plot
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


        # Save scores
        scores = {
            'pred_mae': pred_mae,
            'pred_rmse': pred_rmse,
            'pred_spr': pred_spr,
            'unc_mae': unc_mae,
            'unc_rmse': unc_rmse,
        }
        with open(scores_path, 'w') as f:
          json.dump(scores, f)


        # Plot
        pred_mae = np.array(pred_mae)
        pred_rmse = np.array(pred_rmse)
        pred_spr = np.array(pred_spr)
        unc_mae = np.array(unc_mae)
        unc_rmse = np.array(unc_rmse)

        print("Benchmark complete.")
        txt_accuracy = ''
        txt_accuracy += f"MAE: {np.mean(pred_mae):.3f}±{np.std(pred_mae):.3f}"
        txt_accuracy += "\n"
        txt_accuracy += f"MAE folds:        {np.array2string(pred_mae, precision=3)}"
        txt_accuracy += "\n"
        txt_accuracy += f"MAE unc folds: {np.array2string(unc_mae, precision=3)}"
        txt_accuracy += "\n\n"
        txt_accuracy += f"RMSE: {np.mean(pred_rmse):.3f}±{np.std(pred_rmse):.3f}"
        txt_accuracy += "\n"
        txt_accuracy += f"RMSE folds:        {np.array2string(pred_rmse, precision=3)}"
        txt_accuracy += "\n"
        txt_accuracy += f"RMSE unc folds: {np.array2string(unc_rmse, precision=3)}"
        txt_accuracy += "\n\n"
        txt_accuracy += f"SPEARMAN: {np.mean(pred_spr):.3f}±{np.std(pred_spr):.3f}"
        txt_accuracy += "\n"
        txt_accuracy += f"SPEARMAN folds: {np.array2string(pred_spr, precision=3)}"
        # print(txt_accuracy)

        plt.plot(
            np.linspace(
                np.min(md_t.df_targets.values)-0.5,
                np.max(md_t.df_targets.values)+0.5,
                3,
            ),
            np.linspace(
                np.min(md_t.df_targets.values)-0.5,
                np.max(md_t.df_targets.values)+0.5,
                3,
            ),
            color="black",
            ls="--",
        )

        plt.subplots_adjust(right=0.5)
        plt.ylabel("Predicted $n$")
        plt.xlabel("Computed $n$")
        plt.xlim((np.min(md_t.df_targets.values)-0.5,np.max(md_t.df_targets.values)+0.5))
        plt.ylim((np.min(md_t.df_targets.values)-0.5,np.max(md_t.df_targets.values)+0.5))
        # plt.title(f"MAE: {(scores):.3f}±{np.std(scores):.3f}")
        plt.text(np.max(md_t.df_targets.values)+0.6, 
                 (np.max(md_t.df_targets.values)-np.min(md_t.df_targets.values))/2 + np.min(md_t.df_targets.values),
                 verticalalignment='center',
                 s = txt_accuracy)

        plt.savefig(fig_path)
        
        
        return scores
        


def feat_selec(
        X=None,
        Y=None,
        md_feat=None, 
        feature_selection_kwargs=None,
        md_featselec_dir_path=Path('./moddata_featselec'),
        save=True,
        icycle=0, # index of active learning cycle
        ikf=0     # index of the cross-validation fold
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



def load_feat_selec_df(md_feat_selec, selec=True, target=True):
    assert type(md_feat_selec)==str or type(md_feat_selec)==PosixPath or type(md_feat_selec)==MODData

    if type(md_feat_selec)==str or type(md_feat_selec)==PosixPath:
        md = MODData.load(md_feat_selec)
    else:
        md = deepcopy(md_feat_selec)
    
    X = md.df_featurized

    if selec:
        Y = md.df_targets
        # Let's save the important information of the featurized MODData for easier reuse
        dct_md_info={
            'optimal_features':             md.optimal_features,
            'optimal_features_by_target':   md.optimal_features_by_target,
            'num_classes':                  md.num_classes
        }

        return X, Y, dct_md_info
    elif target and not selec:
        Y = md.df_targets
        return X, Y
    else:
        return X




def predict(
        X, model, results_dir_path = Path('./production/results'),
        icycle=0,
):
    # New MODData to predict
    md_p = MODData(
        materials       = [None]*X.shape[0],
        df_featurized   = X,
        structure_ids   = X.index.values.flatten(),
    )

    if not os.path.isdir(results_dir_path):
        os.makedirs(results_dir_path)
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



def select(Xt, Yt, Xp, Yp, results, acquisition=None, frac=0.05, n=None, **acquisition_kwargs):
    # unpacking the results
    predictions, uncertainties = results
    if not acquisition:
        if frac:
            n=int(len(Xt)*frac)
        id_selected = predictions.sample(n=n, axis=0).index.values
    else:
        scored = acquisition(predictions, uncertainties, **acquisition_kwargs)
        scored = scored.sort_values(by='score',ascending=False).dropna()
        if n:
            id_selected = scored.index.values[:n]
        else:
            id_selected = scored.index.values[:int(len(Xt)*frac)]
        
    
    Xs = Xp.filter(items=id_selected, axis=0)
    Ys = Yp.filter(items=id_selected, axis=0)
    return pd.concat([Xt, Xs], axis=0), pd.concat([Yt, Ys], axis=0), Xp.drop(Xs.index, axis=0), Yp.drop(Ys.index, axis=0)



def train(
    X, Y, dct_md_info,
    model_type=FitGenetic, model_params=None, model_dir_path = (Path(".") / "production" / "models"),
    icycle=0
    ):

    # Create a new training MODData
    md_t = MODData(df_featurized=X)
    md_t.df_targets                 = Y
    md_t.optimal_features           = dct_md_info['optimal_features']
    md_t.optimal_features_by_target = dct_md_info['optimal_features_by_target']
    md_t.num_classes                = dct_md_info['num_classes']

    # Load or train and save
    if not os.path.isdir(model_dir_path):
        os.makedirs(model_dir_path)
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

    return model


def train_pool_random(X, Y, n=None, frac=0.1, state=None, axis=None):
    # Train dataset
    Xt = X.sample(n=n, frac=frac, random_state=state, axis=0)

    Yt = Y.filter(items=Xt.index.values, axis=0)

    # Pool dataset
    Xp = X.drop(Xt.index.values, axis=0)
    Yp = Y.drop(Yt.index.values, axis=0)

    return Xt, Yt, Xp, Yp

    
