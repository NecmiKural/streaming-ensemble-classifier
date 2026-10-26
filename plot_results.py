"""Draws figures/tournament_score.png from results.csv and results-pre-fix.csv: the scores on the
synthetic stream, with the submitted champion (trees never capped) beside the fixed one.

    pip install matplotlib && python3 plot_results.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def synthetic(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["classifier"].split(" ")[0].split("(")[0]: r for r in csv.DictReader(f) if "(synthetic)" in r["classifier"]}


fixed, submitted = synthetic("results.csv"), synthetic("results-pre-fix.csv")
bars = [
    ("Naive Bayes baseline", float(fixed["NaiveBayes"]["tournament_score"]), "#9ca3af"),
    ("Spec ensemble (n=10)", float(fixed["EnsembleClassifier"]["tournament_score"]), "#9ca3af"),
    ("Champion as submitted:\ntrees never capped, over the limit", float(submitted["ChampionClassifier"]["tournament_score"]), "#dc2626"),
    (f"Champion fixed:\n{float(fixed['ChampionClassifier']['memory_kb']):.0f} KB, under the 250 KB budget", float(fixed["ChampionClassifier"]["tournament_score"]), "#2563eb"),
]

fig, ax = plt.subplots(figsize=(8, 3.4))
ys = range(len(bars))[::-1]
drawn = ax.barh(list(ys), [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
ax.bar_label(drawn, fmt="%.4f", fontsize=9, padding=3)
ax.set_yticks(list(ys), [b[0] for b in bars], fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_xlabel("Tournament score (synthetic RBF stream, 100,000 instances)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/tournament_score.png", dpi=180)
