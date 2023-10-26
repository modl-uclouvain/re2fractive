"""
This module defines a set of jobflow workflows that form a
"campaign" for optimising a set of computed properties
across a set of multi-fidelity datasets.

The aim is to use active learning to reduce the number of
expensive high-fidelity calculations required to achieve
an exhaustive screening.

MODNet will be used as the framework for prediction.

"""

from typing import Iterable
from pathlib import Path
import pandas as pd
from re2fractive.campaign import Campaign, Dataset, OptimadeStructure

from atomate2.vasp.jobs.core import DielectricMaker
from modnet.models import EnsembleMODNetModel
from pymatgen.core import Structure


class ModifiedDielectricMaker(DielectricMaker):
    pass


class MP2023Dataset(Dataset):

    properties = {"hull_distance": "_mp_energy_above_hull", "band_gap": "_mp_band_gap"}

    def __init__(self):
        import json

        with open(
            Path(__file__).parent.parent.parent
            / "experiments/data/mp2023_structures.json"
        ) as f:
            structures = json.load(f)

        self.data = [OptimadeStructure(s) for s in structures]


class NaccaratoDataset(Dataset):

    properties = {
        "refractive_index": "_naccarato_refractive_index",
        "band_gap": "_naccarato_gga_bandgap",
        "hull_distance": "_mp_energy_above_hull",
    }

    def __init__(self):
        import json

        with open(
            Path(__file__).parent.parent.parent / "experiments/data/structures.json"
        ) as f:
            structures = json.load(f)

        self.data = [OptimadeStructure(structures[s]).entry.dict() for s in structures]

        # fill in values for hull distance; if its in the MP then its low enough for us for now
        for ind, d in enumerate(self.data):
            self.data[ind]["attributes"]["_mp_energy_above_hull"] = 0


def explore_stable_and_unique(
    predictions: pd.DataFrame, computed: pd.DataFrame
) -> Iterable[Structure]:
    """Given two dataframes of predicted values and computed values,
    with column "structure" and a column per property value and its
    uncertainty, yield which structure in `predictions` to compute next.
    """

    # assumes keys:
    # "structure": pmg.Structure
    # "refractive_index": float
    # "refractive_index_std": float
    # "band_gap": float
    # "hull_distance": float
    candidates = predictions.copy()
    # add hard constraints on e.g., stability and uniqueness
    candidates = candidates[candidates["hull_distance"] < 0.1]

    # Remove duplicate formulae between sets
    computed_formulae = set(computed["structure"].apply(lambda s: s.formula))
    for formula in computed_formulae:
        candidates.drop(
            candidates[
                candidates["structure"].apply(lambda s: s.formula) == formula
            ].index
        )

    yield from candidates["structure"].values.tolist()


DATASET = NaccaratoDataset()


def lookup_refractive_index_oracle(s):
    id = s["id"]
    for d in DATASET.data:
        if d["id"] == id:
            return {"refractive_index": d["attributes"]["_naccarato_refractive_index"]}
    else:
        raise KeyError(f"Could not find id {id} in dataset")


if __name__ == "__main__":
    re2fractive = Campaign(
        properties=["refractive_index", "band_gap", "hull_distance"],
        oracles=[(["refractive_index"], lookup_refractive_index_oracle)],
        datasets=[NaccaratoDataset, MP2023Dataset],
        model=EnsembleMODNetModel,
        logistics={
            "local": False,
            "jfr_project": "re2fractive",
            "jfr_preferred_worker": "lumi",
        },
        learning_strategy={"min_data_points": 100, "min_increment": 5},
        explore_acquisition_function=explore_stable_and_unique,
        acquisition_function=explore_stable_and_unique,
        drop_initial_cols=["refractive_index"],
    )

    re2fractive.run()
    re2fractive.dump_checkpoint("checkpoint.pkl")
