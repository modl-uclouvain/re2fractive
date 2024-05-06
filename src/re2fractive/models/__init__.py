import os

from modnet.hyper_opt import FitGenetic
from modnet.models import EnsembleMODNetModel
from modnet.preprocessing import MODData

from re2fractive import MODELS_DIR
from re2fractive.datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def fit_model(
    dataset: Dataset | MODData | list[Dataset],
    model_cls: type[EnsembleMODNetModel] = EnsembleMODNetModel,
    hyper_opt: bool = False,
    ensemble_n_models: int = 32,
    ensemble_n_feat: int | None = None,
    ensemble_architecture: list[list[int]] | None = None,
    n_jobs: int | None = None,
    bootstrap: bool = True,
) -> tuple[EnsembleMODNetModel, int]:
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    existing_models = [int(m.name) for m in MODELS_DIR.glob("*")]

    model_id: int = 1
    if existing_models:
        model_id = max(existing_models) + 1

    if ensemble_n_feat is None:
        ensemble_n_feat = 32

    if ensemble_architecture is None:
        ensemble_architecture = [[32], [16], [16], [8]]

    moddata = None
    if isinstance(dataset, Dataset):
        moddata = dataset.as_moddata()
        if not getattr(moddata, "optimal_features", None):
            raise RuntimeError(f"Feature selection not performed for {dataset}")
    elif isinstance(dataset, MODData):
        moddata = dataset
    else:
        raise NotImplementedError("Multiple datasets not supported yet")

    if n_jobs is None:
        import multiprocessing as mp

        n_jobs = max(2, mp.cpu_count() - 2)

    if hyper_opt:
        ga = FitGenetic(moddata)
        model = ga.run(
            size_pop=20,
            num_generations=10,
            nested=0,
            n_jobs=n_jobs,
            early_stopping=4,
            refit=ensemble_n_models,
            fast=False,
        )
    else:
        model = model_cls(
            n_models=ensemble_n_models,
            targets=[[list(moddata.target_names)]],
            weights={t: 1.0 for t in list(moddata.target_names)},
            n_feat=ensemble_n_feat,
            num_neurons=ensemble_architecture,
            bootstrap=bootstrap,
        )

        model.fit(moddata, n_jobs=n_jobs)

    (MODELS_DIR / str(model_id)).mkdir(exist_ok=True, parents=True)
    model.save(str(MODELS_DIR / str(model_id) / f"{model_id}.pkl"))
    moddata.df_targets.to_csv(MODELS_DIR / str(model_id) / "training.csv")

    return model, model_id


def load_model(model_id: int):
    if (MODELS_DIR / str(model_id) / f"{model_id}.pkl").exists():
        model = EnsembleMODNetModel.load(
            str(MODELS_DIR / str(model_id) / f"{model_id}.pkl")
        )
        return model
    raise FileNotFoundError(f"Model {model_id} not found")
