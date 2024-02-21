import json
from pathlib import Path
from modnet.featurizers.presets import Matminer2023Featurizer

import tqdm
from modnet.preprocessing import MODData
from optimade.adapters import Structure
from optimade.models.utils import reduce_formula

N_CPU = 6

data_path = Path(__file__).absolute().parent.parent / "data"
db_path = data_path / "db.csv"
structures_path = data_path / "structures-2024.json"
moddata_path = Path("./mod.data")
feature_selected_path = Path("./mod-feature-selected.data")

if not moddata_path.exists():

    with open(structures_path) as f:
        structures = json.load(f)

    pmg_structures = []
    targets = []

    for ind, s in tqdm.tqdm(enumerate(structures.items())):
        s[1]["attributes"]["chemical_formula_reduced"] = reduce_formula(
            s[1]["attributes"]["chemical_formula_reduced"]
        )
        targets.append(
            s[1]["attributes"]["_naccarato_refractive_index"],
        )
        if s[0] == "mp-505722":
            breakpoint()
        pmg_structures.append(Structure(s[1]).as_pymatgen)

    moddata = MODData(
        materials=pmg_structures,
        targets=targets,
        featurizer=Matminer2023Featurizer(fast_oxid=True),
        target_names=["refractive_index"],
        structure_ids=structures.keys(),
    )
    moddata.featurizer.featurizer_mode = "single"
    moddata.featurize(n_jobs=N_CPU)

    moddata.save("mod.data")

if not feature_selected_path.exists():
    moddata = MODData.load("mod.data")
    moddata.feature_selection(n=-1, n_jobs=N_CPU)
    moddata.save("mod-feature-selected.data")
