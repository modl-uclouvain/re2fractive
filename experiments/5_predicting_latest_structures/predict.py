from pathlib import Path
import numpy as np
import pandas as pd
import os
from modnet.preprocessing import MODData
from modnet.models import EnsembleMODNetModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

moddata_path = Path(__file__).parent.parent / "data" / "mod.data_feature_selected"
feature_store = (
    Path(__file__).parent.parent / "data" / "mp_2023_df_featurized_multi.pkl"
)
models = Path(__file__).parent.parent / "models"

moddata = MODData.load(moddata_path)
featurized_df = pd.read_pickle(feature_store)
moddata.df_featurized = featurized_df
# MODData Hack to get around the fact that the structure IDs have also changed
moddata.df_structure = featurized_df

results_dfs = {}
for i in range(5):
    model = EnsembleMODNetModel.load(models / f"baseline_kfold_ensemble-refractive_index-tdpsw_{i}.pkl")
    results_dfs[i] = model.predict(moddata)

results_df = results_dfs[0].copy()
# results_df["optical_gap_std"] = np.nan
results_df["refractive_index_std"] = np.nan
for j in range(len(results_dfs[0])):
    # results_df.iloc[j, results_df.columns.get_loc("optical_gap")] = np.mean(
    #     [results_dfs[i].iloc[j]["optical_gap"] for i in range(5)]
    # )
    # results_df.iloc[j, results_df.columns.get_loc("optical_gap_std")] = np.std(
    #     [results_dfs[i].iloc[j]["optical_gap"] for i in range(5)]
    # )
    results_df.iloc[j, results_df.columns.get_loc("refractive_index")] = np.mean(
        [results_dfs[i].iloc[j]["refractive_index"] for i in range(5)]
    )
    results_df.iloc[j, results_df.columns.get_loc("refractive_index_std")] = np.std(
        [results_dfs[i].iloc[j]["refractive_index"] for i in range(5)]
    )

results_df.to_pickle("results.pkl")
