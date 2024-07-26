"""This module defines the core logic of a re2fractive `Campaign`.

A campaign is defined by target property(s), a set of datasets,
oracles, model and featurization parameters and a learning strategy.

"""

import datetime
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Literal, TypeAlias

import pandas as pd
from jobflow import Maker
from modnet.models import EnsembleMODNetModel, MODNetModel
from modnet.preprocessing import MODData
from sklearn.model_selection import train_test_split

from re2fractive import EPOCHS_DIR, RESULTS_DIR
from re2fractive.acquisition import extremise_expected_value, random_selection
from re2fractive.acquisition.optics import from_w_eff
from re2fractive.datasets import Dataset
from re2fractive.featurizers import BatchableMODFeaturizer, MatminerFastFeaturizer
from re2fractive.models import fit_model, load_model


@dataclass
class CampaignLogistics:
    """A collection of settings related to deploying the campaign."""

    local: bool = True
    jfr_project: str | None = None
    jfr_preferred_worker: str | None = None


@dataclass
class Epoch:
    """A container for the results from each epoch."""

    model_metrics: dict = field(default_factory=dict)
    """The metrics of the model on the holdout set."""

    model_id: int | None = None
    """The ID of the model that was trained in this epoch."""

    design_space: list[pd.DataFrame] | None = None
    """The design space that was evaluated in this epoch."""

    selected: pd.DataFrame | None = None
    """The selected results from the design space at this epoch."""


@dataclass
class LearningStrategy:
    """A collection of settings that define the active learning strategy for the campaign."""

    initial_val_fraction: float = field(default=0.2)
    """The fraction of the initial dataset to use as a holdout set throughout."""

    random_seed: int = field(default=42)
    """A random seed that can be controlled to make campaign runs deterministic."""

    min_data_points: int = field(default=100)
    """The minimum number of data points required before training a model."""

    min_increment: int = field(default=5)
    """The minimum number of data points required since the last training before
    refitting the model.
    """

    min_hyperopt_increment: int = field(default=500)
    """The minimum number of data points required between full
    hyperparameter optimisation runs.
    """

    feature_select_strategy: Literal["once", "always"] = field(default="once")
    """Whether to perform feature selection on the datasets at each epoch, or just the first."""

    max_n_features: int | None = None
    """An upper bound to add to the number of features that can be used in the model."""

    model_n_jobs: int | None = None
    """The number of processes to use when training models."""

    hyperopt_always: bool = field(default=True)
    """Whether to run hyperparameter optimisation on every model refit."""

    hyperopt_strategy: Literal["once", "never", "always"] = field(default="always")
    """The strategy to use for hyperparameter optimisation.

        - If 'once', hyperparameter optimisation will be run once at the start of the campaign.
        - If 'never', hyperparameter optimisation will never be run, instead using the default parameters specified here.
        - If 'always', hyperparameter optimisation will be run on every model refit.

    """

    bootstrap: bool = field(default=True)
    """Whether to bootstrap sample the datasets when using an EnsembleMODnetModel."""

    ensemble_n_models: int = field(default=32)
    """How many models to train in each ensemble."""

    ensemble_n_feat: int | None = None
    """The number of features to use in each ensemble (ignored when doing hyperparameter optimisation)."""

    ensemble_architecture: list[list[int]] | None = None
    """The architecture of the ensemble model to use (ignored when doing hyperparameter optimisation)."""

    acquisition_function: Callable = from_w_eff
    """The acquisition function to use for selecting new trials.
    Will receive a list of dataframes of the design space, and
    is expected to return a subset or sorted dataframe of the
    materials to select.

    """


Oracle: TypeAlias = Callable | Maker


@dataclass
class Campaign:
    """The container class for the active learning campaign that stores
    references to results and the current state of the campaign.

    """

    oracles: list[tuple[tuple[str, ...], Oracle]]
    """A list of oracles that can be evaluated to obtain the 'ground truth', and
    the associated properites they can compute.

    These should be provided in the form of jobflow `Maker`'s that can be
    excuted remotely.

    """

    properties: list[str]
    """A list of properties that are either available or computable by the oracles,
    that can be used in acquisition functions.
    """

    model_cls: type[MODNetModel]
    """The type of model to train and use for prediction."""

    datasets: list[Dataset]
    """A list of datasets to be used in the campaign.

    These will be used in turn to train surrogate models and
    explore the space.

    """

    logistics: CampaignLogistics = field(default_factory=CampaignLogistics)
    """A series of settings used to execute the jobflow workflows."""

    learning_strategy: LearningStrategy = field(default_factory=LearningStrategy)
    """The active learning strategy to take for training the models."""

    models: list = field(default_factory=list)
    """A list of models that have been trained so far."""

    explore_acquisition_function: Callable = random_selection
    """An explore-stage acqusition function that can e.g., weight
    property uncertainty vs. constraints to improve model performance
    or simply explore the space.
    """

    acquisition_function: Callable = extremise_expected_value
    """An exploit-stage acqusition function that defines the desired
    screening wrt. final constraints and optimisation targets. """

    epochs: list[Epoch] = field(default_factory=list)
    """A container that stores the epochs that have already been run."""

    train_moddata: MODData = None
    """The training data that is used to train the model at the current campaign state."""

    campaign_uuid: str | None = None
    """A UUID that uniquely identifies this campaign."""

    featurizer: BatchableMODFeaturizer = field(default=MatminerFastFeaturizer)
    """The featurizer to use during learning."""

    @classmethod
    def new_campaign_from_dataset(
        cls,
        initial_dataset: Dataset | type[Dataset],
        datasets: list[type[Dataset]] | None = None,
        oracles: dict[str, Oracle] | None = None,
        learning_strategy: LearningStrategy = LearningStrategy(),
    ):
        """Initialise a new campaign from some given datasets."""
        from re2fractive import CAMPAIGN_ID

        if CAMPAIGN_ID is not None:
            campaign_uuid = CAMPAIGN_ID
        else:
            campaign_uuid = "0001"

        if isinstance(initial_dataset, type):
            loaded_dataset = initial_dataset.load()  # type: ignore
            if loaded_dataset is not None:
                initial_dataset = loaded_dataset
        assert isinstance(initial_dataset, Dataset)
        assert initial_dataset is not None
        return cls(
            oracles=oracles if oracles is not None else [],  # type:ignore
            properties=list(initial_dataset.properties.keys()),
            model_cls=EnsembleMODNetModel,
            learning_strategy=learning_strategy,
            datasets=[type(initial_dataset)] + datasets
            if datasets
            else [type(initial_dataset)],
            campaign_uuid=campaign_uuid,
        )

    def first_step(self, model_id: int | None = None):
        """Kick off the first steps of a campaign by taking the
        defined `initial_dataset`, separating out a holdout set
        according to the `learning_strategy`, and training a model
        on it, which is then applied to all remaining datasets.

        Any datasets that need to be featurized along the way will be.

        """

        initial_dataset = self.datasets[0].load()
        initial_dataset.featurize_dataset(
            self.featurizer,
            feature_select=True,
            max_n_features=self.learning_strategy.max_n_features,
        )

        # featurize other datasets
        if len(self.datasets) > 1:
            for d in self.datasets[1:]:
                d = d.load()
                d.featurize_dataset(self.featurizer, feature_select=False)

        train_inds, test_inds = train_test_split(
            range(len(initial_dataset)),
            test_size=self.learning_strategy.initial_val_fraction,
            random_state=self.learning_strategy.random_seed,
        )

        assert len(train_inds) > len(test_inds)

        # Initialise the global holdout set and the initial training data
        self._holdout_inds = test_inds
        self.train_moddata, _ = initial_dataset.as_moddata().split(
            (train_inds, test_inds)
        )
        model_id, holdout_metrics, design_space = self.learn_and_evaluate(
            model_id=1,
        )

        self.finalize_epoch(holdout_metrics, model_id, design_space)

    def finalize_epoch(self, holdout_metrics, model_id, design_space, results_df=None):
        epoch = {
            "model_metrics": holdout_metrics,
            "model_id": model_id,
            "design_space": [d.to_dict(orient="index") for d in design_space],
            "selected": results_df.to_dict(orient="index")
            if results_df is not None
            else None,
        }

        self.epochs.append(epoch)
        self.checkpoint()

    def learn_and_evaluate(self, model_id: int | None = None):
        if model_id is not None:
            model = load_model(model_id)
        else:
            print("Fitting new model")
            model, model_id = fit_model(
                self.train_moddata,
                n_jobs=self.learning_strategy.model_n_jobs,
                bootstrap=self.learning_strategy.bootstrap,
                hyper_opt=self.learning_strategy.hyperopt_always,
                ensemble_n_models=self.learning_strategy.ensemble_n_models,
                ensemble_n_feat=self.learning_strategy.ensemble_n_feat,
                ensemble_architecture=self.learning_strategy.ensemble_architecture,
            )

        print(f"Evaluating global holdout {model_id=}")
        _, _, _, metrics, _ = self.evaluate_global_holdout(model)

        print(f"Evaluating design space with {model_id=}")
        predictions, std_devs = self.evaluate_design_space(model)

        return model_id, metrics, (predictions, std_devs)

    def checkpoint(self) -> None:
        import json
        import pickle

        last_epoch = str(len(self.epochs) - 1)
        last_epoch_dir = EPOCHS_DIR / last_epoch
        last_epoch_dir.mkdir(exist_ok=True, parents=True)
        if (last_epoch_dir / f"{last_epoch}.json").exists():
            raise RuntimeError("Found existing epoch file, aborting checkpoint.")
        with open(last_epoch_dir / f"{last_epoch}.json", "w") as f:
            json.dump(self.epochs[-1], f)

        with open(last_epoch_dir / "campaign.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, epoch: int | None = None) -> "Campaign":
        import pickle

        if epoch is None:
            epoch_names = sorted(list(EPOCHS_DIR.iterdir()))
            final_epoch = epoch_names[-1]
        else:
            final_epoch = EPOCHS_DIR / str(epoch)
        with open(final_epoch / "campaign.pkl", "rb") as f:
            campaign = pickle.load(f)

        return campaign

    def evaluate_global_holdout(self, model: EnsembleMODNetModel):
        """Make predictions with a given model on the global holdout set
        from the initial dataset.

        """
        initial_dataset = self.datasets[0].load()
        holdout_set = initial_dataset.as_moddata().from_indices(self._holdout_inds)

        preds, stds = model.predict(
            holdout_set, return_unc=True, remap_out_of_bounds=True
        )

        errors = holdout_set.df_targets - preds
        mae = errors.abs().mean().values[0]
        metrics = {
            "mean_absolute_error": float(mae),
            "mean_uncertainty": float(stds.mean().values[0]),
        }

        print("Holdout set metrics:")
        pprint(metrics)

        return preds, errors, stds, metrics, holdout_set

    def evaluate_design_space(self, model: EnsembleMODNetModel) -> list[pd.DataFrame]:
        """Make predictions with a given model on the design space
        of the remaining datasets.

        Returns a list of dataframes, each containing the predictions
        for each dataset.

        """
        if len(self.datasets) <= 1:
            raise RuntimeError("No datasets left to evaluate.")
        design_spaces = [d.load() for d in self.datasets[1:]]

        results = []

        for d in design_spaces:
            print(f"Evaluating {d.__class__.__name__}")
            preds, stds = model.predict(
                d.as_moddata(), return_unc=True, remap_out_of_bounds=True
            )
            col = stds.columns[0]
            stds.rename(columns={col: f"{col}_std"}, inplace=True)

            results.append(pd.concat([preds, stds], axis=1))

        return results

    def run(self, epochs: int = 1, wait: bool = True) -> None:
        """March the campaign for a number of epochs.

        If wait is `True`, after potentially submitting calculations, this function will keep
        polling the filesystem until they have been completed. If interrupted, re-running
        the function will begin polling again.

        """
        for _ in range(epochs):
            self.march(wait=wait)

    def march(self, wait: bool = True) -> None:
        """Marches the campaign forward through the next step, based on the current state.

        Each epoch will *end* with the latest prediction of the design space. Each new epoch
        starts by selecting new trials for computation, and then deciding whether to train or
        update the model.

        If wait is `True`, after potentially submitting calculations, this function will keep
        polling the filesystem until they have been completed. If interrupted, re-running
        the function will begin polling again.

        """

        if not self.epochs:
            return self.first_step()

        # Check whether an epoch was submitted previously, load from it if so
        # Find only epochs that have a campaign.pkl
        this_epoch_index: int = max(
            [int(d.parent.name) for d in EPOCHS_DIR.glob("*/campaign.pkl")]
        )

        # If campaign.pkl exists, this epoch is already run
        if (EPOCHS_DIR / str(this_epoch_index) / "campaign.pkl").exists():
            this_epoch_index += 1
            self.start_new_epoch()

        print(f"Polling epoch {this_epoch_index}")
        self.poll_epoch(this_epoch_index, wait=wait)

        print(f"Gathering results for epoch {this_epoch_index}")
        results_df = self.gather_results(this_epoch_index)

        print(f"Gathering features for epoch {this_epoch_index}")
        featurized_df, target_df = self.gather_features(results_df)
        self.update_training_moddata(featurized_df, target_df)

        print(f"Retraining model for {this_epoch_index}")
        model_id, holdout_metrics, design_space = self.learn_and_evaluate(
            model_id=this_epoch_index
        )
        self.finalize_epoch(holdout_metrics, model_id, design_space, results_df)

    def start_new_epoch(self):
        design_space = self.epochs[-1]["design_space"]
        ranking = self.make_selection(design_space)

        # # Submit or get pre-computed trials from database
        # new_calcs = self.submit_oracle(ranking)

    def _epoch_finished(self, epoch_index: int) -> bool:
        """Check the epoch dir for results from all calculations."""
        epoch_calc_dir = EPOCHS_DIR / str(epoch_index) / "calcs"
        if not epoch_calc_dir.exists():
            return False

        print(f"Found calc dir for epoch {epoch_index}")
        return True

        # For now, we assume that all calculations are finished as we are
        # manually populating the calcs dirs

        expected_calc_ids = {
            d["id"]
            for d in json.loads(
                Path(epoch_calc_dir / "_submitted_ids.json").read_text()
            )
        }
        finished_calc_ids = {
            str(d.name.split(".")[0]) for d in epoch_calc_dir.glob("[A-z,0-9]*.json")
        }
        return expected_calc_ids == finished_calc_ids

    def gather_features(
        self, results_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loop over a set of oracle results and find the featurized
        and target data for the corresponding structures to be used in the next
        epoch's training set.

        Returns:
            The featurized df and the target df.

        """
        featurized_results_df = pd.DataFrame()
        target_df = pd.DataFrame()
        datasets = [d.load() for d in self.datasets[1:]]

        print("Gathering results")
        for d in datasets:
            # Collect the featurized data for the structures in the results
            moddata = d.as_moddata()
            print(f"Gathering results from {d.__class__.__name__}...")
            features = moddata.df_featurized[
                moddata.df_featurized.index.isin(results_df.index)
            ]
            print(f"Found {features.shape} features.")
            featurized_results_df = pd.concat(
                [
                    featurized_results_df,
                    features,
                ]
            )
            target_df = pd.concat(
                [target_df, results_df[results_df.index.isin(features.index)]]
            )

        print(
            f"Gathered {featurized_results_df.shape} features for {results_df.shape} results."
        )

        return featurized_results_df, target_df

    def gather_results(self, epoch_index: int) -> pd.DataFrame | None:
        """Gather results from the calculations in the epoch dir.

        Any folder that is written inside the `calcs` dir will be treated
        as a selected calculation. This means that auxiliary calculations
        that were not automatically selected can also be provided to the model
        at the next training run.

        Returns the results dataframe for the chosen epoch of calcs, if it exists.

        """
        epoch_calc_dir = EPOCHS_DIR / str(epoch_index) / "calcs"
        results = []
        for calc_dir in epoch_calc_dir.glob("*"):
            results_file = calc_dir / "result.json"
            if results_file.exists():
                results.append(json.loads(results_file.read_text()))

        if not results:
            return None

        return pd.DataFrame(results).set_index("id")

    def update_training_moddata(self, featurized_results, target_results):
        print(
            f"Updating training moddata... previously {self.train_moddata.df_featurized.shape}"
        )
        all_featurized_dfs = pd.concat(
            [self.train_moddata.df_featurized, featurized_results],
        )
        all_targets = pd.concat([self.train_moddata.df_targets, target_results])[
            self.datasets[0].targets
        ].values
        print(f"Updated training moddata... now {all_featurized_dfs.shape}")
        moddata = MODData(
            df_featurized=all_featurized_dfs,
            targets=all_targets,
            target_names=self.train_moddata.target_names,
            structure_ids=all_featurized_dfs.index.values,
        )

        if self.learning_strategy.feature_select_strategy == "always":
            print("Redoing feature selection according to strategy...")
            n = self.learning_strategy.max_n_features
            if n is None:
                n = -1
            moddata.feature_selection(n=n, drop_thr=0.05)

        return moddata

    def poll_epoch(self, epoch_index: int, wait: bool = True) -> bool:
        """Look up in the calculations list whether this calcs of this
        calcs have finished/terminated, and update the epoch accordingly.

        Returns `True` once the epoch has completed, `False` otherwise.

        """

        while True:
            if self._epoch_finished(epoch_index):
                return True
            if wait:
                print(
                    f"{datetime.datetime.now()} -- epoch {epoch_index} not yet finished, waiting 10 minutes."
                )
                time.sleep(600)
                continue
            else:
                print(
                    f"{datetime.datetime.now()} -- epoch {epoch_index} not yet finished, exiting..."
                )
                return False

    def parcel_up_structures(self) -> None:
        """Saves all of the structures used in the campaign to a folder of CIF files,
        with associated .csv containing any computed properties.

        """

        from optimade.adapters.structures import Structure

        if not RESULTS_DIR.exists():
            RESULTS_DIR.mkdir()

        if not (RESULTS_DIR / "structures").exists():
            (RESULTS_DIR / "structures").mkdir()

        results = []
        for epoch_ind, epoch in enumerate(self.epochs):
            r = self.gather_results(epoch_ind)
            if r is not None:
                results.append(r)

        pd.concat(results).to_csv(RESULTS_DIR / "results.csv")

        print("Loading datasets...")
        for ind, dataset in enumerate(self.datasets):
            self.datasets[ind] = dataset.load()

        for epoch in results:
            for i, row in epoch.iterrows():
                structure = None
                for dataset in self.datasets:
                    # find structure
                    try:
                        structure = dataset.structure_df.loc[i]["structure"]
                    except Exception:
                        continue

                if not structure:
                    try:
                        structure = Structure.from_url(i).as_pymatgen
                    except Exception as e:
                        print(
                            f"Bad structure {i}: not in original dataset and entry download failed with message: {e}"
                        )
                        continue

                id = i.split("/")[-1]
                structure.to(filename=RESULTS_DIR / "structures" / f"{id}.cif")

    def make_selection(self, design_space):
        return self.learning_strategy.acquisition_function(design_space)
