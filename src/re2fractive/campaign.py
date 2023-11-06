import os
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from jobflow import Flow, Maker, run_locally
from modnet.models import MODNetModel
from optimade.adapters import Structure as OptimadeStructure

from re2fractive.datasets import Dataset
from re2fractive.selection import extremise_expected_value, random_selection


@dataclass
class OptimadeQuery:
    filter: str
    """The OPTIMADE filter to be used to query the databases."""

    providers: list[str]
    """The list of database providers to query. Can point to either an index meta-database
    or an individual OPTIMADE API.

    """


@dataclass
class CampaignLogistics:
    local: bool
    jfr_project: str | None
    jfr_preferred_worker: str | None


@dataclass
class LearningStrategy:
    min_data_points: int = field(default=100)
    """The minimum number of data points required before training a model."""

    min_increment: int = field(default=5)
    """The minimum number of data points required since the last training before
    refitting the model.
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

    logistics: CampaignLogistics
    """A series of settings used to execute the jobflow workflows."""

    learning_strategy: LearningStrategy
    """The active learning strategy to take for training the models."""

    models: list = field(default_factory=list)
    """A list of models that have been trained so far."""

    drop_initial_cols: list[str] | None = None
    """For testing, drop this column from the initial dataset."""

    initial_model: MODNetModel | None = None
    """An initial model to use to generate the first round of predictions."""

    explore_acquisition_function: Callable = random_selection
    """An explore-stage acqusition function that can e.g., weight
    property uncertainty vs. constraints to improve model performance
    or simply explore the space.
    """

    acquisition_function: Callable = extremise_expected_value
    """An exploit-stage acqusition function that defines the desired
    screening wrt. final constraints and optimisation targets.
    """

    epochs: list[dict] = []
    """A container that stores the epochs that have already been run."""

    _campaign_uuid: str | None = None
    """A UUID that uniquely identifies this campaign."""

    def __post_init__(self):
        self.logistics = CampaignLogistics(**self.logistics)  # type: ignore
        self.learning_strategy = LearningStrategy(**self.learning_strategy)  # type: ignore
        if not self._campaign_uuid:
            import uuid

            self._campaign_uuid = str(uuid.uuid4())

        self.models = []
        if self.initial_model:
            self.models.append(self.initial_model)

    @staticmethod
    def load_checkpoint(fname: os.PathLike) -> "Campaign":
        import pickle

        with open(fname, "rb") as f:
            return pickle.load(f)

    @property
    def oracle_property_map(self) -> dict[str, Oracle]:
        map = {}
        for oracle_desc in self.oracles:
            properties = oracle_desc[0]
            for p in properties:
                map[p] = oracle_desc[1]
        return map

    def dump_checkpoint(self, fname: Path | None = None) -> None:
        import pickle

        if fname is None:
            fname = Path(
                f"campaign_checkpoint-{self._campaign_uuid}-{len(self.epochs)}.pkl"
            )

        i = 0
        while fname.exists():
            fname = Path(str(fname).replace(".pkl", f"-{i}.pkl"))
            i += 1

        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def load_initial_dataset(self):
        """Load the initial dataset from the provided datasets."""

        dataset = self.datasets[0]
        dataset.load()

        property_counts = {}
        for property in self.properties:
            property_name_in_dataset = dataset.properties.get(property)
            if property_name_in_dataset is None:
                property_counts[property] = 0
                continue
            if self.drop_initial_cols and property in self.drop_initial_cols:
                print(f"Dropping {property}")
                for ind, _ in enumerate(dataset.data):
                    dataset.data[ind]["attributes"].pop(property_name_in_dataset)

            property_counts[property] = sum(
                d["attributes"].get(property_name_in_dataset) is not None
                for d in dataset.data
            )

        return dataset, property_counts

    # def prepare_model_training(self, dataset, property_counts):

    #     for p in property_counts:
    #         if property_counts[p] < self.learning_strategy.min_data_points:
    #             if p not in self.oracle_property_map:
    #                 print(
    #                     f"Cannot get more values for property {p}, no avaialble oracle from {self.oracle_property_map=}"
    #                 )
    #             elif self.learning_strategy.min_data_points > len(dataset.data):
    #                 print(
    #                     f"Dataset {dataset=} not large enough to train model with strategy {self.learning_strategy.min_data_points=}"
    #                 )
    #             else:
    #                 # start by randomly computing some values
    #                 # in reality these should be saved in a database
    #                 for ind, d in enumerate(dataset.data):
    #                     if d["attributes"].get(dataset.properties[p]) is not None:
    #                         continue
    #                     oracle = self.oracle_property_map[p]
    #                     print(f"Computing {p} for {d['id']}")
    #                     dataset.data[ind]["attributes"][dataset.properties[p]] = oracle(
    #                         d
    #                     )[p]

    def train_initial_model(self, dataset):
        """Train an initial model on the initial dataset."""
        from modnet.preprocessing import MODData

        training_data_subset = [
            d
            for d in dataset.data
            if d["attributes"].get("refractive_index") is not None
        ]
        training_data = MODData(
            [OptimadeStructure(d).as_pymatgen for d in training_data_subset],
            [d["attributes"]["refractive_index"] for d in training_data_subset],
        )

        training_data.featurize(n_jobs=1)

        model = self.model_cls(
            targets=[["refractive_index"]], weights={"refractive_index": 1}
        )

        model.fit(training_data)

        self.models.append(model)

        return model

    def run_epoch(self):
        dataset_counter = 0
        if self.models:
            model = self.models[-1]
        else:
            raise RuntimeError(
                "No trained models yet; please initialise campaign with existing model or train one."
            )

        self.predict(model, self.datasets[dataset_counter])
        candidates = self.select(self.datasets[dataset_counter])
        workflows: dict[str, dict[tuple[str, ...], Any]] = {}

        for candidate in candidates:
            workflows[candidate.id] = {}
            for prop, oracle in self.oracles:
                if isinstance(oracle, Maker):
                    compute = oracle.make
                else:
                    compute = oracle
                workflows[candidate.id][prop] = compute(candidate)

        epoch = {
            "candidates": candidates,
            "workflows": workflows,
            "dataset_counter": dataset_counter,
        }
        self.epochs.append(epoch)

        for candidate in candidates:
            for prop, oracle in self.oracles:
                self.evaluate_workflow(workflows[candidate.id][prop])

        # get the property values out of the workflow
        for candidate in candidates:
            for prop, oracle in self.oracles:
                candidate[prop] = workflows[candidate.id][prop].output

        epoch["computed_candidates"] = candidates
        self.epochs[-1].update(epoch)

        # Update the dataset(s) with an inefficient loop for now
        candidate_ids = {candidate.id for candidate in candidates}
        for ind, d in enumerate(self.datasets[dataset_counter]):
            if d.id in candidate_ids:
                self.datasets[dataset_counter][ind].update(candidate)

        # Then potentially retrain the model
        self.dump_checkpoint()

    def evaluate_workflow(self, workflow: Flow):
        if self.logistics.local:
            run_locally(workflow)

    def exploit_explore_heuristic(self):
        return random.random() < 0.5

    def select(self, dataset):
        """Select the next structures to evaluate."""
        if self.exploit_explore_heuristic():
            selector = self.explore_acquisition_function
        else:
            selector = self.explore_acquisition_function

        return selector(
            [d for d in dataset if d.get(property) is None],
            dataset,
            num_to_select=self.learning_strategy.min_increment,
        )

    def predict(self, model: MODNetModel, candidates: Iterable[OptimadeStructure]):
        """Massage dataset into format for the model and run prediction step."""
        import pandas as pd

        df_featurized = pd.DataFrame([d.attributes for d in candidates])
        return model.predict(df_featurized)

    def run_setup(self):
        _ = self.load_initial_dataset()

        if self.initial_model is None:
            raise RuntimeError("No initial model provided")
            # self.models.append(self.prepare_model_training(dataset, property_counts))

        self.models.append(self.initial_model)
