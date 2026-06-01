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

def calc_is2(x):
    x = np.asarray(x, dtype=float)
    m = np.nanargmin(x)
    xmin = np.nanmean(x[max(m - 5, 0):min(m + 6, len(x))])
    xf = np.nanmean(x[-60:])
    return (x - xmin) / abs(xf - xmin)

def time_to_level(time, signal, level, start=0):
    idx = start + np.nanargmin(np.abs(signal[start:] - level))
    return (time[idx] - time[0]) / 60.0

groups = []

for (gen, trans), g in df.groupby(["Genotype", "TransientType"]):
    g = g.sort_values("Time").copy()

    if trans in ["transient1", "transient2", "transient4"]:
        g["IS"] = calc_is(g["sPhoto"])
        g["simIS"] = calc_is(g["simPhoto"])
    else:
        g["IS"] = calc_is2(g["sPhoto"])
        g["simIS"] = calc_is2(g["simPhoto"])

    groups.append(g)

df2 = pd.concat(groups)

up_rows = []
down_rows = []

for (gen, trans), g in df2.groupby(["Genotype", "TransientType"]):
    g = g.sort_values("Time")
    time = g["Time"].to_numpy()
    IS = g["IS"].to_numpy()
    simIS = g["simIS"].to_numpy()

    if trans in ["transient1", "transient2", "transient4"]:
        up_rows.append({
            "Genotype": gen,
            "TransientType": trans,
            "t30": time_to_level(time, IS, 0.3),
            "simt30": time_to_level(time, simIS, 0.3),
            "t50": time_to_level(time, IS, 0.5),
            "simt50": time_to_level(time, simIS, 0.5),
            "t70": time_to_level(time, IS, 0.7),
            "simt70": time_to_level(time, simIS, 0.7),
            "t90": time_to_level(time, IS, 0.9),
            "simt90": time_to_level(time, simIS, 0.9),
        })

    if trans in ["transient3", "transient5"]:
        m_obs = np.nanargmin(IS)
        m_sim = np.nanargmin(simIS)

        row = {
            "Genotype": gen,
            "TransientType": trans,
            "tmin": (time[m_obs] - time[0]) / 60.0,
            "simtmin": (time[m_sim] - time[0]) / 60.0,
        }

        for lvl in [0.5, 0.7, 0.9]:
            key = int(lvl * 100)
            row[f"t{key}"] = time_to_level(time, IS, lvl, start=m_obs)
            row[f"simt{key}"] = time_to_level(time, simIS, lvl, start=m_sim)

        down_rows.append(row)

table_up = pd.DataFrame(up_rows)
table_down = pd.DataFrame(down_rows)

colors = {
    "30": "black",
    "50": "red",
    "70": "limegreen",
    "90": "blue",
    "min": "black",
}

markers = {
    "col": "o",
    "npq1": "^",
    "npq4": "+",
    "spsa": "x",
    "rwt43": "D",
    "rca2": "v",
}

genotypes = ["col", "npq1", "npq4", "spsa", "rwt43", "rca2"]

panel_info = {
    "transient1": ("A", "0 → 1000"),
    "transient2": ("B", "70 → 800"),
    "transient4": ("C", "130 → 600"),
}

fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.2))
fig.patch.set_facecolor("white")
axes = axes.flatten()

def style_axis(ax):
    ax.set_facecolor("white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 30)
    ax.set_ylim(1, 30)

    ticks = [1, 2, 3, 5, 10, 20, 30]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])

    ax.plot([1, 30], [1, 30], color="black", linewidth=1.1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(direction="out", length=5, width=1, colors="black")

def plot_point(ax, x, y, gen, color):
    if np.isnan(x) or np.isnan(y):
        return

    marker = markers[gen]

    if marker in ["+", "x"]:
        ax.scatter(
            x, y,
            marker=marker,
            color=color,
            s=50,
            linewidths=1.25,
        )
    else:
        ax.scatter(
            x, y,
            marker=marker,
            edgecolors=color,
            facecolors="none",
            s=55,
            linewidths=1.25,
        )

def legend_marker_row(ax, y, color):
    xs = [1.45, 1.67, 1.92, 2.17, 2.45, 2.72]
    for x, gen in zip(xs, genotypes):
        plot_point(ax, x, y, gen, color)

def add_custom_legend(ax, first_metric="30"):
    if first_metric == "min":
        first_text = (
            r"$t_{\min}$ (Col-0, $npq1$-2, $npq4$-1,"
            "\n"
            r"$spsa1$, $rwt43$, $rca$-2)"
        )
    else:
        first_text = (
            r"$t_{30}$ (Col-0, $npq1$-2, $npq4$-1,"
            "\n"
            r"$spsa1$, $rwt43$, $rca$-2)"
        )

    yvals = [26.0, 20.0, 16.0, 13.0]
    labels = [
        first_text,
        r"$t_{50}$ (idem)",
        r"$t_{70}$ (idem)",
        r"$t_{90}$ (idem)",
    ]
    cols = ["black", "red", "limegreen", "blue"]

    for y, label, col in zip(yvals, labels, cols):
        legend_marker_row(ax, y, col)
        ax.text(2.95, y, label, color=col, fontsize=7.2, va="center")

for ax, trans in zip(axes[:3], ["transient1", "transient2", "transient4"]):
    sub = table_up[table_up["TransientType"] == trans]

    for gen in genotypes:
        sg = sub[sub["Genotype"] == gen]
        if sg.empty:
            continue

        row = sg.iloc[0]

        for metric in ["30", "50", "70", "90"]:
            plot_point(
                ax,
                row[f"t{metric}"],
                row[f"simt{metric}"],
                gen,
                colors[metric],
            )

    style_axis(ax)
    ax.text(1.05, 27.0, panel_info[trans][0], fontsize=13, va="top")
    ax.text(5.0, 1.22, panel_info[trans][1], ha="center", fontsize=11)
    add_custom_legend(ax, first_metric="30")

ax = axes[3]

for gen in genotypes:
    sg = table_down[table_down["Genotype"] == gen]
    if sg.empty:
        continue

    for _, row in sg.iterrows():
        plot_point(ax, row["tmin"], row["simtmin"], gen, colors["min"])

        for metric in ["50", "70", "90"]:
            plot_point(
                ax,
                row[f"t{metric}"],
                row[f"simt{metric}"],
                gen,
                colors[metric],
            )

style_axis(ax)
ax.text(1.05, 27.0, "D", fontsize=13, va="top")
ax.text(5.0, 1.55, "800 → 130", ha="center", fontsize=11)
ax.text(5.0, 1.23, "600 → 200", ha="center", fontsize=11)
add_custom_legend(ax, first_metric="min")

fig.supxlabel("Measured time index (min)", fontsize=14)
fig.supylabel("Simulated time index (min)", fontsize=14)

plt.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.10,
    top=0.98,
    wspace=0.00,
    hspace=0.00,
)

out = PYTHON_FIGURES / "Figure3_python_paper_style.png"
plt.savefig(out, dpi=1000, facecolor="white")
plt.show()

print("Saved:", out)