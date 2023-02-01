"""Downloads the Naccarato refractive index dataset,
grabs the corresponding Materials Project structures through
OPTIMADE then saves them locally.

"""

from pathlib import Path
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
    silent=True,
)

structures = {}
if structures_path.exists():
    with open(structures_path, "r") as f:
        structures = json.load(f)

for ind, row in tqdm.tqdm(df.iterrows(), ):

    mp_id = row["MP_id"]
    if mp_id in structures:
        continue

    filter_ = f'id = "{mp_id}"'
    
    try:
        structure = client.get(filter_)["structures"][filter_]["https://optimade.materialsproject.org"]["data"][0]
        structures[mp_id] = structure
    except Exception:
        print(f"No structure found for {mp_id}")
    if ind % 10 == 0:
        with open(structures_path, "w") as f:
            json.dump(structures, f, indent=2)
