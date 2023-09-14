"""Download the latest MP structures that obey the filters
from the previous dataset.

"""

from pathlib import Path
import json
import os
import pandas as pd
from mp_api.client import MPRester
from modnet.featurizers.presets import Matminer2023Featurizer
from optimade.adapters import Structure


data_path = Path(__file__).parent.parent / "data" / "mp2023_structures.json"

if not data_path.exists():
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        docs = mpr.summary.search(
            band_gap=(0.3, None),
            energy_above_hull=(0, 0.025),
        )

    optimade_docs = []

    for doc in docs:
        structure = Structure.ingest_from(doc.structure)
        optimade_doc = structure.as_dict
        optimade_doc["attributes"]["_mp_band_gap"] = doc.band_gap
        optimade_doc["attributes"]["_mp_energy_above_hull"] = (doc.energy_above_hull,)
        optimade_doc["attributes"]["structure_origin"] = (
            "experimental" if not doc.theoretical else "predicted"
        )
        optimade_doc["attributes"][
            "_mp_formation_energy_per_atom"
        ] = doc.formation_energy_per_atom
        optimade_doc["attributes"]["immutable_id"] = str(doc.material_id)
        optimade_doc["id"] = str(doc.material_id)
        optimade_docs.append(optimade_doc)

    with open(data_path, "w") as f:
        json.dump(optimade_docs, f)

else:
    with open(data_path, "r") as f:
        optimade_docs = json.load(f)


df_structures = pd.DataFrame.from_dict(
    {doc["id"]: {"structure": Structure(doc).as_pymatgen} for doc in optimade_docs}, orient="index"
)



featurizer = Matminer2023Featurizer()
featurizer.oxid_composition_featurizers = []
featurizer.featurizer_mode = "multi"
featurizer.set_n_jobs(8)

df_featurized = featurizer.featurize(df_structures)
df_featurized.to_pickle(data_path.parent / "mp_2023_df_featurized_multi.pkl")
