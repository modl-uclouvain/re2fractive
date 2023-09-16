import pandas as pd
from pathlib import Path
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
palette = plt.get_cmap('tab10').colors

df = pd.read_pickle("results_df.pkl")

mp_structures_file = Path(__file__).parent.parent / "data"/ "mp2023_structures.json"
with open(mp_structures_file) as f:
    mp_structures = json.load(f)

# Add DFT band gaps to predicted values
df["band_gap"] = -1.0
df["formula"] = ""
df["nelements"] = 0
for doc in tqdm.tqdm(mp_structures):
    id = doc["id"]
    df["band_gap"].loc[id] = doc["attributes"]["_mp_band_gap"]
    df["formula"].loc[id] = doc["attributes"]["chemical_formula_reduced"]
    df["nelements"].loc[id] = doc["attributes"]["nelements"]

fig, ax = plt.subplots()
with open("../data/structures.json") as f:
    original_structures = json.load(f)

# drop duplicates by ID between both sets
print(f"Initial set: {len(df)}")
df = df.drop(original_structures.keys(), errors="ignore")
print(f"After de-duplication by ID: {len(df)}")

# Drop formula duplicates
formulae = set(original_structures[doc]["attributes"]["chemical_formula_reduced"] for doc in original_structures)
for formula in formulae:
    df.drop(df[df["formula"] == formula].index[1:], inplace=True)
print(f"After de-duplication by formula: {len(df)}")

ax.errorbar(df["band_gap"], df["refractive_index"], yerr=df["refractive_index_std"], alpha=0.1, c=palette[0], fmt="o", marker=None)
ax.scatter(df["band_gap"], df["refractive_index"], s=10, c=palette[0], alpha=0.5, label="MP 2023 ($E_g$ > 0.3 eV, $E_{hull}$ < 0.025 eV/atom)")

ax.scatter(
    [doc["attributes"]["_naccarato_gga_bandgap"] for doc in original_structures.values()],
    [doc["attributes"]["_naccarato_refractive_index"] for doc in original_structures.values()],
    c=palette[1],
    alpha=1,
    zorder=1000,
    s=3,
    edgecolor="w",
    lw=0.2,
    label="Training set"
)

potential_candidates = []
candidates = df[df["band_gap"] * df["refractive_index"] > 10][df["refractive_index"] > 2][df["band_gap"] < 15][df["nelements"] > 1]
for row in candidates.itertuples():
    ax.annotate(row.Index, (row.band_gap, row.refractive_index), fontsize=12)
candidates.to_csv("candidates.csv")

ax.plot(np.linspace(0, 15, 100), 3.3668 * np.linspace(0, 15, 100) ** -0.32234, c="gold", lw=2, label="Kumar & Singh", zorder=1e5)
ax.legend()
ax.set_xlabel("Optical gap (eV)")
ax.set_ylabel("Refractive index")

plt.savefig("n_vs_gap.png", dpi=300, facecolor="w")
