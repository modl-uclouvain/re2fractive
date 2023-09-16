import os
import string
import random
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    import tensorflow as tf
    targets = ["refractive_index"]
    experiment_name = "baseline_kfold_ensemble-" + "-".join(targets) + "-" + "".join(random.choice(string.ascii_lowercase) for _ in range(5))

    gpus = tf.config.list_physical_devices("GPU")
    gpus = None
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    from modnet.models import EnsembleMODNetModel
    from modnet.preprocessing import MODData

    data: MODData = MODData.load("mod.data_feature_selected")

# scrub now missing features
    bad_features = [
        'AtomicPackingEfficiency|dist from 1 clusters |APE| < 0.010',
        'ElectronegativityDiff|maximum EN difference',
        'ElectronegativityDiff|mean EN difference',
        'AtomicPackingEfficiency|dist from 3 clusters |APE| < 0.010',
        'AtomicPackingEfficiency|mean simul. packing efficiency',
        'ElectronegativityDiff|range EN difference',
        'ElectronegativityDiff|minimum EN difference'
    ]
    data.optimal_features = [col for col in data.optimal_features if col not in bad_features]
    for k in data.optimal_features_by_target:
        data.optimal_features_by_target[k] = [col for col in data.optimal_features_by_target[k] if col not in bad_features]

    kf = KFold(5, shuffle=True, random_state=42)
    scores = []

    for ind, (train, test) in enumerate(
        kf.split(data.df_featurized, y=data.df_targets)
    ):
        train_moddata, test_moddata = data.split((train, test))
        model_path = (
            Path(__file__).parent.parent / "models" / f"{experiment_name}_{ind}.pkl"
        )
        if model_path.exists():
            model = EnsembleMODNetModel.load(model_path)

        else:
            model = EnsembleMODNetModel(
                targets=[[["refractive_index"]]],
                weights={"refractive_index": 1.0},
                num_neurons=([128], [32], [32], [16]),
                n_feat=64,
                n_models=32,
            )

            model.fit(
                train_moddata,
                n_jobs=12,
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
            np.min(data.df_targets.values),
            np.max(data.df_targets.values),
            3,
        ),
        np.linspace(
            np.min(data.df_targets.values),
            np.max(data.df_targets.values),
            3,
        ),
        color="black",
        ls="--",
    )

    plt.ylabel("Predicted $n$ (dimensionless)")
    plt.xlabel("Computed $n$ (dimensionless)")
    plt.savefig(Path(__file__).parent.parent / "figs" / f"{experiment_name}.pdf")
