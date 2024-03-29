import abc
import datetime
import json
import os
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from optimade.adapters.structures import Structure as OptimadeStructure
from optimade.client import OptimadeClient

from re2fractive import CAMPAIGN_DIR, DATASETS_DIR


class Dataset(abc.ABC):
    """The Dataset object provides a container for OPTIMADE structures
    that are decorated with the same set of properties.

    """

    id: str
    """A tag for the dataset, e.g. "Naccarato2019" or "MP2023"."""

    references: list[dict] | None
    """Bibliographic references for the dataset, if available."""

    metadata: dict
    """Any additional metadata for the dataset, will be saved in the dataset directory as meta.json"""

    data: list[OptimadeStructure]
    """A list of OPTIMADE structures, decorated with target properties, where available."""

    properties: dict[str, str]
    """A dictionary mapping from a property name to the column name in the dataset."""

    def __init__(self):
        if getattr(self, "id", None) is None:
            self.id = self.__class__.__name__.replace("Dataset", "")

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def as_df(self):
        df = pd.DataFrame(
            [{"id": entry.id, **entry.as_dict["attributes"]} for entry in self.data]
        )
        return df.set_index("id")

    @classmethod
    def load(cls, campaign_id: str = "0001") -> "Dataset | None":
        filename = (
            CAMPAIGN_DIR / campaign_id / DATASETS_DIR / f"{cls.id}" / f"{cls.id}.jsonl"
        )
        self = cls()

        if filename.exists():
            with open(filename) as f:
                self.data = [OptimadeStructure(json.loads(s)) for s in f.readlines()]

            return self

        return None

    def save(self, campaign_id: str = "0001"):
        dataset_dir = CAMPAIGN_DIR / campaign_id / DATASETS_DIR / f"{self.id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(dataset_dir / f"{self.id}.jsonl", "w") as f:
            for entry in self.data:
                f.write(entry.as_json + "\n")

        with open(dataset_dir / "meta.json", "w") as f:
            json.dump(self.metadata, f)


class MP2023Dataset(Dataset):
    properties = {"hull_distance": "_mp_energy_above_hull", "band_gap": "_mp_band_gap"}

    id: str = "MP2023"

    @classmethod
    def load(cls, campaign_id: str = "0001") -> "MP2023Dataset":
        """Use the MP API to load a dataset of materials with a band gap and energy above hull,
        then convert these to the OPTIMADE format and alias some properties.

        """

        self = super().load(campaign_id)
        if self is not None:
            return self  # type: ignore

        print(
            f"Previously created dataset not found; loading {cls.id} dataset from scratch"
        )

        from mp_api.client import MPRester

        with MPRester(os.environ["MP_API_KEY"]) as mpr:
            docs = mpr.summary.search(
                band_gap=(0.05, None),
                energy_above_hull=(0, 0.025),
            )

        optimade_docs = []

        for doc in docs:
            structure = OptimadeStructure.ingest_from(doc.structure)
            optimade_doc = structure.as_dict
            optimade_doc["attributes"]["_mp_band_gap"] = doc.band_gap
            optimade_doc["attributes"]["_mp_energy_above_hull"] = (
                doc.energy_above_hull,
            )
            optimade_doc["attributes"]["_mp_structure_origin"] = (
                "experimental" if not doc.theoretical else "predicted"
            )
            optimade_doc["attributes"]["_mp_formation_energy_per_atom"] = (
                doc.formation_energy_per_atom
            )
            optimade_doc["attributes"]["immutable_id"] = str(doc.material_id)
            optimade_doc["attributes"]["_mp_refractive_index"] = doc.n
            optimade_doc["id"] = str(doc.material_id)
            optimade_docs.append(optimade_doc)

        self = cls()
        self.data = [OptimadeStructure(doc) for doc in optimade_docs]
        self.metadata = {"ctime": datetime.datetime.now().isoformat()}
        self.save(campaign_id)
        return self


class NaccaratoDataset(Dataset):
    properties = {
        "refractive_index": "_naccarato_refractive_index",
        "band_gap": "_naccarato_gga_bandgap",
        "hull_distance": "_mp_energy_above_hull",
    }

    id: str = "Naccarato2019"

    references: list[dict] | None = [
        {
            "authors": ["Naccarato, F.", "others"],
            "doi": "10.1103/PhysRevMaterials.3.044602",
        }
    ]

    @classmethod
    def load(cls, campaign_id: str = "0001") -> "NaccaratoDataset":
        self = super().load(campaign_id)
        if self is not None:
            return self  # type: ignore

        print(
            f"Previously created dataset not found; loading {cls.id} dataset from scratch"
        )

        db_path = CAMPAIGN_DIR / campaign_id / DATASETS_DIR / cls.id / "Naccarato.csv"

        if not db_path.exists():
            urllib.request.urlretrieve(
                "https://journals.aps.org/prmaterials/supplemental/10.1103/PhysRevMaterials.3.044602/db.csv",
                db_path,
            )

        df = pd.read_csv(db_path)

        self = cls()

        print(f"Initial number in {self.id}: {len(df)}")

        structures = []

        client = OptimadeClient("https://optimade.materialsproject.org", silent=True)
        for _, row in tqdm.tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Downloading MP structures matching {self.id}",
        ):
            mp_id = row["MP_id"]
            if mp_id in structures:
                continue

            filter_ = f'id = "{mp_id}"'
            structure: dict[str, dict[str, Any]] = {}
            structure["attributes"] = {}
            try:
                structure = client.get(filter_)["structures"][filter_][
                    "https://optimade.materialsproject.org"
                ]["data"][0]
            except Exception:
                print(f"No structure for {mp_id}")
                continue

            if structure["id"] != mp_id:
                print(f"ID mismatch: {structure['id']} != {mp_id}")
                continue

            structure["attributes"]["_naccarato_refractive_index"] = np.mean(
                [row["Ref_index (1)"], row["Ref_index (2)"], row["Ref_index (3)"]]
            )
            structure["attributes"]["_naccarato_average_optical_gap"] = row[
                "Average_optical_gap (eV)"
            ]
            structure["attributes"]["_naccarato_gga_bandgap"] = row[
                "GGA direct band gap (eV)"
            ]
            structure["attributes"]["_naccarato_effective_frequency"] = row[
                "Effective_frequency (eV)"
            ]

            # For some reason, some structures do not have the right stability info through OPTIMADE, so we assume it is 0
            try:
                structure["attributes"]["_mp_energy_above_hull"] = structure[
                    "attributes"
                ]["_mp_stability"]["gga_gga+u_r2scan"]["energy_above_hull"]
            except KeyError:
                structure["attributes"]["_mp_energy_above_hull"] = 0.0

            structures.append(structure)

        self.data = [OptimadeStructure(s) for s in structures]
        print(f"Final number in {self.id}: {len(structures)}")

        self.metadata = {"ctime": datetime.datetime.now().isoformat()}

        self.save(campaign_id)
        return self
