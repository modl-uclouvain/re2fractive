"""Downloads the Naccarato refractive index dataset,
grabs the corresponding Materials Project structures through
OPTIMADE then saves them locally.

"""

from pathlib import Path
import contextlib
import json

from optimade.client import OptimadeClient
import pandas as pd
import tqdm

data_path = Path(__file__).parent.parent / "data"
db_path = data_path / "db.csv"
structures_path = data_path / "structures.json"

if not db_path.exists():
    import urllib.request
    urllib.request.urlretrieve(
        "https://journals.aps.org/prmaterials/supplemental/10.1103/PhysRevMaterials.3.044602/db.csv", 
        db_path,
    )

df = pd.read_csv(db_path)

client = OptimadeClient(
    base_urls="https://optimade.materialsproject.org",
)

structures = {}
if structures_path.exists():
    with open(structures_path, "r") as f:
        structures = json.load(f)

for ind, row in tqdm.tqdm(df.iterrows(), total=len(df)):

    mp_id = row["MP_id"]
    if mp_id in structures:
        continue

    filter_ = f'id = "{mp_id}"'
    
    try:
        # redirect output from stdout to dev null
        with contextlib.redirect_stdout(None) and contextlib.redirect_stderr(None):
            structure = client.get(filter_)["structures"][filter_]["https://optimade.materialsproject.org"]["data"][0]
    except Exception:
        print(f"No structure found for {mp_id}")
    structure["attributes"]["_naccarato_refractive_index"] = row["Ref_index (1)"]
    structures[mp_id] = structure
    if ind % 100 == 0:
        with open(structures_path, "w") as f:
            json.dump(structures, f, indent=2)
