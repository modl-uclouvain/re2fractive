import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
palette = plt.get_cmap('tab10').colors

df = pd.read_pickle("results_df.pkl")

fig, ax = plt.subplots()
with open("../data/structures.json") as f:
    original_structures = json.load(f)


# drop duplicates between both sets
print(f"Initial set: {len(df)}")
df = df.drop(original_structures.keys(), errors="ignore")
print(f"After de-duplication: {len(df)}")

ax.errorbar(df["optical_gap"], df["refractive_index"], xerr=df["optical_gap_std"], yerr=df["refractive_index_std"], alpha=0.1, c=palette[0], fmt="o", marker=None)
ax.scatter(df["optical_gap"], df["refractive_index"], s=10, c=palette[0], alpha=0.5, label="MP 2023 ($E_g$ > 0.3 eV, $E_{hull}$ < 0.025 eV/atom)")

ax.scatter(
    [doc["attributes"]["_naccarato_average_optical_gap"] for doc in original_structures.values()],
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
candidates = df[df["optical_gap"] * df["refractive_index"] > 23.5][df["refractive_index"] > 2]
for row in candidates.itertuples():
    ax.annotate(row.Index, (row.optical_gap, row.refractive_index), fontsize=12)
candidates.to_csv("candidates.csv")

ax.plot(np.linspace(0, 15, 100), 3.3668 * np.linspace(0, 15, 100) ** -0.32234, c="gold", lw=2, label="Kumar & Singh", zorder=1e5)
ax.legend()
ax.set_xlabel("Optical gap (eV)")
ax.set_ylabel("Refractive index")

plt.savefig("n_vs_gap.pdf")
