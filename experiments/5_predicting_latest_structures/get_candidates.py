from mp_api.client import MPRester
import pandas as pd
import os
import monty
import json

candidates = pd.read_csv("candidates.csv", index_col=0)

ids = candidates.index.to_list()

with MPRester(os.environ["MP_API_KEY"]) as mpr:
    docs = mpr.summary.search(material_ids=ids)

structures = [doc.structure for doc in docs]

with open("candidate_structures.json", "w") as f:
    json.dump(structures, f, cls=monty.json.MontyEncoder)
