import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paths import R_OUTPUT, PYTHON_FIGURES

df = pd.read_csv(R_OUTPUT / "transients_with_simTransients.csv")

def calc_is(x):
    x = np.asarray(x, dtype=float)
    xA = np.nanmean(x[:60])
    xB = np.nanmean(x[-60:])
    return (x - min(xA, xB)) / abs(xB - xA)

def time_to_level(time, signal, level):
    idx = np.nanargmin(np.abs(signal - level))
    return (time[idx] - time[0]) / 60.0

groups = []

for (gen, trans), g in df.groupby(["Genotype", "TransientType"]):
    g = g.sort_values("Time").copy()
    g["ISgs"] = calc_is(g["sCond"])
    g["simISgs"] = calc_is(g["simCond"])
    groups.append(g)

df2 = pd.concat(groups)

rows = []

for (gen, trans), g in df2[df2["TransientType"].isin(
    ["transient1", "transient2", "transient3", "transient4"]
)].groupby(["Genotype", "TransientType"]):

    g = g.sort_values("Time")
    time = g["Time"].to_numpy()
    ISgs = g["ISgs"].to_numpy()
    simISgs = g["simISgs"].to_numpy()

    row = {
        "Genotype": gen,
        "TransientType": trans,
        "t30": time_to_level(time, ISgs, 0.3),
        "simt30": time_to_level(time, simISgs, 0.3),
        "t50": time_to_level(time, ISgs, 0.5),
        "simt50": time_to_level(time, simISgs, 0.5),
        "t70": time_to_level(time, ISgs, 0.7),
        "simt70": time_to_level(time, simISgs, 0.7),
        "t90": time_to_level(time, ISgs, 0.9),
        "simt90": time_to_level(time, simISgs, 0.9),
    }

    rows.append(row)

table = pd.DataFrame(rows)

# R note: "There is a crazy t90 value of 0.2 minutes..."
table = table[table["t90"] > 1].copy()

colors = {
    "30": "black",
    "50": "red",
    "70": "limegreen",
    "90": "blue",
}

# R symbols = transient type number 1 to 4
markers = {
    "transient1": "o",
    "transient2": "^",
    "transient3": "+",
    "transient4": "x",
}

fig, ax = plt.subplots(figsize=(4.0, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for _, row in table.iterrows():
    marker = markers[row["TransientType"]]

    for metric in ["30", "50", "70", "90"]:
        x = row[f"t{metric}"]
        y = row[f"simt{metric}"]

        if np.isnan(x) or np.isnan(y):
            continue

        if marker in ["+", "x"]:
            ax.scatter(
                x,
                y,
                marker=marker,
                color=colors[metric],
                s=36,
                linewidths=1.0,
            )
        else:
            ax.scatter(
                x,
                y,
                marker=marker,
                edgecolors=colors[metric],
                facecolors="none",
                s=36,
                linewidths=1.0,
            )

ax.plot([0, 60], [0, 60], linestyle="--", color="black", linewidth=1)

ax.set_xlim(0, 60)
ax.set_ylim(0, 60)

ax.set_xlabel("Measured time index (min)")
ax.set_ylabel("Simulated time index (min)")

def legend_marker_row(y, color):
    xs = [15, 20, 25, 30]
    for x, trans in zip(xs, ["transient1", "transient2", "transient3", "transient4"]):
        marker = markers[trans]
        if marker in ["+", "x"]:
            ax.scatter(x, y, marker=marker, color=color, s=34, linewidths=1.0)
        else:
            ax.scatter(x, y, marker=marker, edgecolors=color, facecolors="none", s=34, linewidths=1.0)

legend_marker_row(55, "black")
legend_marker_row(48, "red")
legend_marker_row(41, "limegreen")
legend_marker_row(34, "blue")

ax.text(34, 55, r"$t_{30}$ (0 - 1000, 70 - 800", fontsize=7.5, va="center")
ax.text(34, 51, r"      800 - 130, 130 - 600)", fontsize=7.5, va="center")
ax.text(34, 48, r"$t_{50}$ (idem)", color="red", fontsize=7.5, va="center")
ax.text(34, 41, r"$t_{70}$ (idem)", color="limegreen", fontsize=7.5, va="center")
ax.text(34, 34, r"$t_{90}$ (idem)", color="blue", fontsize=7.5, va="center")

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(direction="out", length=4, width=1, colors="black")

plt.tight_layout()

out = PYTHON_FIGURES / "figureMeasuredModelledgs_python.png"
plt.savefig(out, dpi=1000, facecolor="white")
plt.show()

print("Saved:", out)