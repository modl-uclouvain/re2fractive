from dataclasses import dataclass, field
import pandas as pd
import os
import random
from jobflow_remote.jobs.state import JobState
from modnet.models import MODNetModel
from jobflow import Maker, Job, Flow, run_locally
from typing import Annotated, Callable, TypeAlias
from optimade.adapters import Structure as OptimadeStructure
from jobflow_remote import JobController, submit_flow


@dataclass
class Stage:
    """A stage in the campaign that need only be run once."""

    factory: Job | Flow
    uuid: str | None = None

    @property
    def status(self) -> JobState:
        """Return the status of the stage."""
        return (
            JobController()
            .get_job_info(
                job_id=self.uuid,
            )
            .state
        )


@dataclass
class OptimadeQuery:

    filter: str
    """The OPTIMADE filter to be used to query the databases."""

    providers: list[str]
    """The list of database providers to query. Can point to either an index meta-database
    or an individual OPTIMADE API.

    """


@dataclass
class Dataset:

    data: list[OptimadeStructure]
    """A list of OPTIMADE structures, decorated with target properties, where available."""

    properties: dict[str, str]
    """A dictionary mapping from a property name to the column name in the dataset."""


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
    oracles: list[tuple[list[str], Oracle]]
    """A list of oracles that can be evaluated to obtain the 'ground truth', and
    the associated properites they can compute.

    These should be provided in the form of jobflow `Maker`'s that can be
    excuted remotely.

    """

    properties: list[str]
    """A list of properties that are either available or computable by the oracles,
    that can be used in acquisition functions.
    """

    model: MODNetModel
    """The type of model to train and use for prediction."""

    datasets: list[Dataset]
    """A list of datasets to be used in the campaign.

    These will be used in turn to train surrogate models and
    explore the space.

    """

    explore_acquisition_function: Callable
    """An explore-stage acqusition function that can e.g., weight
    property uncertainty vs. constraints to improve model performance
    or simply explore the space.
    """

    acquisition_function: Callable
    """An exploit-stage acqusition function that defines the desired
    screening wrt. final constraints and optimisation targets.
    """

    logistics: CampaignLogistics
    """A series of settings used to execute the jobflow workflows."""

    learning_strategy: LearningStrategy
    """The active learning strategy to take for training the models."""

    drop_initial_cols: list[str]
    """For testing, drop this column from the initial dataset."""

    def __post_init__(self):
        self.logistics = CampaignLogistics(**self.logistics)
        self.learning_strategy = LearningStrategy(**self.learning_strategy)

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

    def dump_checkpoint(self, fname: os.PathLike) -> None:
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def load_initial_dataset(self):
        """Load the initial dataset from the provided datasets."""

        dataset = self.datasets[0]()

        property_counts = {}
        for property in self.properties:
            property_name_in_dataset = dataset.properties.get(property)
            if property_name_in_dataset is None:
                property_counts[property] = 0
                continue
            if self.drop_initial_cols and property in self.drop_initial_cols:
                print(f"Dropping {property}")
                for ind, d in enumerate(dataset.data):
                    dataset.data[ind]["attributes"].pop(property_name_in_dataset)

            property_counts[property] = sum(
                d["attributes"].get(property_name_in_dataset) != None
                for d in dataset.data
            )

        return dataset, property_counts

    def prepare_model_training(self, dataset, property_counts):

        for p in property_counts:
            if property_counts[p] < self.learning_strategy.min_data_points:
                if p not in self.oracle_property_map:
                    print(
                        f"Cannot get more values for property {p}, no avaialble oracle from {self.oracle_property_map=}"
                    )
                elif self.learning_strategy.min_data_points > len(dataset.data):
                    print(
                        f"Dataset {dataset=} not large enough to train model with strategy {self.learning_strategy.min_data_points=}"
                    )
                else:
                    # start by randomly computing some values
                    # in reality these should be saved in a database
                    for ind, d in enumerate(dataset.data):
                        if d["attributes"].get(dataset.properties[p]) is not None:
                            continue
                        oracle = self.oracle_property_map[p]
                        print(f"Computing {p} for {d['id']}")
                        dataset.data[ind]["attributes"][dataset.properties[p]] = oracle(
                            d
                        )[p]

    def train_initial_model(self, dataset):
        """Train an initial model on the initial dataset."""
        from modnet.preprocessing import MODData

        training_data_subset = [
            d
            for d in dataset.data
            if d["attributes".get("refractive_index") is not None]
        ]
        training_data = MODData(
            [OptimadeStructure(d).as_pymatgen for d in training_data],
            [d["attributes"]["refractive_index"] for d in training_data],
        )

        training_data.featurize(n_jobs=1)

        model = self.model(
            targets=[["refractive_index"]], weights={"refractive_index": 1}
        )

        model.fit(training_data)

        return model

    def run(self):
        dataset, property_counts = self.load_initial_dataset()
        self.prepare_model_training(dataset, property_counts)
