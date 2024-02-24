from mp_api.client import MPRester
import os
from pathlib import Path
from optimade.adapters import Structure
import json

data_path = Path(__file__).parent.parent / "data" / "mp2024_structures.json"
with MPRester(os.environ["MP_API_KEY"]) as mpr:
    docs = mpr.summary.search(
        band_gap=(0.3, None),
        energy_above_hull=(0, 0.05),
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
