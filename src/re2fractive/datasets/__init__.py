import abc
from pathlib import Path

from optimade.adapters.structures import Structure as OptimadeStructure


class Dataset(abc.ABC):
    """The Dataset object provides a container for OPTIMADE structures
    that are decorated with the same set of properties.

    """

    data: list[OptimadeStructure]
    """A list of OPTIMADE structures, decorated with target properties, where available."""

    properties: dict[str, str]
    """A dictionary mapping from a property name to the column name in the dataset."""

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load(self):
        raise NotImplementedError


class MP2023Dataset(Dataset):
    properties = {"hull_distance": "_mp_energy_above_hull", "band_gap": "_mp_band_gap"}

    def load(self):
        import json

        with open(
            Path(__file__).parent.parent.parent.parent
            / "experiments/data/mp2023_structures.json"
        ) as f:
            structures = json.load(f)

        self.data = structures

        for ind, _ in enumerate(self.data):
            for key, column in self.properties.items():
                self.data[ind]["attributes"][key] = self.data[ind]["attributes"].get(
                    column
                )

        self.data = [OptimadeStructure(s) for s in self.data]


class NaccaratoDataset(Dataset):
    properties = {
        "refractive_index": "_naccarato_refractive_index",
        "band_gap": "_naccarato_gga_bandgap",
        "hull_distance": "_mp_energy_above_hull",
    }

    def load(self):
        import json

        with open(
            Path(__file__).parent.parent.parent.parent
            / "experiments/data/structures.json"
        ) as f:
            structures = json.load(f)

        self.data = [OptimadeStructure(structures[s]).entry.dict() for s in structures]

        # fill in values for hull distance; if it's in the MP then it's low enough for us for now
        for ind, _ in enumerate(self.data):
            self.data[ind]["attributes"]["_mp_energy_above_hull"] = 0
