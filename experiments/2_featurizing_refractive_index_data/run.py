import json
from pathlib import Path

import pandas as pd
import tqdm
from modnet.preprocessing import MODData
from optimade.adapters import Structure
from optimade.models.utils import reduce_formula 

data_path = Path(__file__).absolute().parent.parent / "data"
db_path = data_path / "db.csv"
structures_path = data_path / "structures.json"

with open(structures_path) as f:
    structures = json.load(f)

pmg_structures = []
targets = []

for ind, s in tqdm.tqdm(enumerate(structures.items())):
    s[1]["attributes"]["chemical_formula_reduced"] = reduce_formula(s[1]["attributes"]["chemical_formula_reduced"])
    targets.append((s[1]["attributes"]["_naccarato_refractive_index"], s[1]["attributes"]["_naccarato_optical_gap"]))
    pmg_structures.append(Structure(s[1]).as_pymatgen)
    
moddata = MODData(
    materials=pmg_structures,
    targets=targets,
    target_names=["refractive_index", "optical_gap"],
    structure_ids=structures.keys(),
)
moddata.featurizer.featurizer_mode = "single"
moddata.featurize(n_jobs=1)

moddata.save("mod.data")
