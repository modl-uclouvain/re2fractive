"""This module defines the core logic of a re2fractive `Campaign`.

A campaign is defined by target property(s), a set of datasets,
oracles, model and featurization parameters and a learning strategy.

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pprint import pprint
from typing import TypeAlias

from jobflow import Maker
from modnet.models import EnsembleMODNetModel, MODNetModel
from sklearn.model_selection import train_test_split

from re2fractive import EPOCHS_DIR
from re2fractive.acquisition import extremise_expected_value, random_selection
from re2fractive.datasets import Dataset
from re2fractive.featurizers import BatchableMODFeaturizer, MatminerFastFeaturizer
from re2fractive.models import load_model


@dataclass
class CampaignLogistics:
    local: bool = True
    jfr_project: str | None = None
    jfr_preferred_worker: str | None = None


@dataclass
class LearningStrategy:
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


Oracle: TypeAlias = Callable | Maker


@dataclass
class Campaign:
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

    # drop_initial_cols: list[str] | None = None
    # """For testing, drop this column from the initial dataset."""

    # initial_model: MODNetModel | None = None
    # """An initial model to use to generate the first round of predictions."""

    explore_acquisition_function: Callable = random_selection
    """An explore-stage acqusition function that can e.g., weight
    property uncertainty vs. constraints to improve model performance
    or simply explore the space.
    """

    acquisition_function: Callable = extremise_expected_value
    """An exploit-stage acqusition function that defines the desired
    screening wrt. final constraints and optimisation targets. """

    epochs: list[dict] = field(default_factory=list)
    """A container that stores the epochs that have already been run."""

    campaign_uuid: str | None = None
    """A UUID that uniquely identifies this campaign."""

    featurizer: BatchableMODFeaturizer = field(default=MatminerFastFeaturizer)
    """The featurizer to use during learning."""

    @classmethod
    def new_campaign_from_dataset(
        cls,
        initial_dataset: Dataset | type[Dataset],
        datasets: list[type[Dataset]] | None = None,
    ):
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
            oracles=[],
            properties=list(initial_dataset.properties.keys()),
            model_cls=EnsembleMODNetModel,
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
        from re2fractive.models import fit_model

        initial_dataset = self.datasets[0].load()
        initial_dataset.featurize_dataset(self.featurizer)

        train_inds, test_inds = train_test_split(
            range(len(initial_dataset)),
            test_size=self.learning_strategy.initial_val_fraction,
            random_state=self.learning_strategy.random_seed,
        )

        assert len(train_inds) > len(test_inds)
        self._holdout_inds = test_inds

        train_moddata, _ = initial_dataset.as_moddata().split((train_inds, test_inds))

        if model_id is not None:
            model = load_model(model_id)
        else:
            model, model_id = fit_model(train_moddata)

        _, _, _, metrics, _ = self.evaluate_global_holdout(model)

        predictions, std_devs = self.evaluate_design_space(model)

        epoch = {
            "model_metrics": metrics,
            "model_id": model_id,
            "design_space": {"predictions": predictions, "std_devs": std_devs},
        }

        self.epochs = [epoch]

        self.checkpoint()

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
            holdout_set, return_unc=True, remap_out_of_bounds=False
        )

        errors = holdout_set.df_targets - preds
        mae = errors.abs().mean().values[0]
        metrics = {
            "mean_absolute_error": float(mae),
            "mean_uncertainty": float(stds.mean().values[0]),
        }

        pprint(metrics)

        return preds, errors, stds, metrics, holdout_set

    def evaluate_design_space(self, model: EnsembleMODNetModel):
        """Make predictions with a given model on the design space
        of the remaining datasets.

        """
        if len(self.datasets) <= 1:
            raise RuntimeError("No datasets left to evaluate.")
        design_spaces = [d.load() for d in self.datasets[1:]]

        predictions = []
        std_devs = []

        for d in design_spaces:
            preds, stds = model.predict(
                d.as_moddata(), return_unc=True, remap_out_of_bounds=False
            )

            predictions.append(preds.values.tolist())
            std_devs.append(stds.values.tolist())

        return predictions, std_devs

    def march(self):
        """Marches the campaign forward through the next step, based on the current state."""

        if not self.epochs:
            return self.first_step()
