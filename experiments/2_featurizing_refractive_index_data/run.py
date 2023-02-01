import json
from pathlib import Path
import pandas as pd

data_path = Path(__file__).parent.parent / "data"
db_path = data_path / "db.csv"
structures_path = data_path / "structures.json"

with open(structures_path) as f:
    structures = json.load(f)

breakpoint()

from modnet.preprocessing import MODData
moddata = MODData(
    structures,
    targets=pd.read_csv(db_path)["Ref_index (1)"],
    target_names=["n"],
    structure_ids=structures.keys(),
)

moddata.featurize()
