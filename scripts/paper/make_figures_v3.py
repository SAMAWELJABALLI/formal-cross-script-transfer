"""V3 embedding figures from reports/v3/{viz_coords,geometry_v3}.json (dev rows only; real cached embeddings).
Usage:  python scripts/paper/make_figures_v3.py   (writes to $FORMAL_OUT/figures and $FORMAL_OUT/reports/v3/axes_stats.json)
  fig_beforeafter_pca.pdf   fixed PCA basis (fitted on raw dev embeddings), mE5 / LaBSE / XLM-R x 4 conditions
  fig_pca3d.pdf             real 3-D PCA (fixed basis), mE5, raw vs human-pair projection
  fig_diagnostics.pdf       script probe, pair retrieval, held-out residual energy, overlap with human subspace
  figS1_tsne.pdf, figS2_umap.pdf, figS3_pca3d_all.pdf, figS5_tsac_pca.pdf (supplementary)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGURES, REPORTS_V3, prefer_out

FIG = FIGURES
V = json.load(open(prefer_out("reports/v3/viz_coords.json")))
G = json.load(open(prefer_out("reports/v3/geometry_v3.json")))
BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6})
ENCN = {"e5-small": "mE5-small", "labse": "LaBSE", "xlmr-base": "XLM-R"}
SRCN = {"none": "no projection", "human": "human pairs", "rule": "rule pairs", "dict": "dictionary pairs",
        "neural": "neural pairs", "tarc": "TArC pairs"}
lab = np.array(V["labels"]); n = len(lab)
script = np.array(["ar"] * n + ["az"] * n); sent = np.concatenate([lab, lab])
COL = {"neg": BLUE, "pos": ORANGE}


def scatter2(ax, C, s=14, lines=True):
    C = np.asarray(C)
    if lines:
        for i in range(n):
            ax.plot([C[i, 0], C[n + i, 0]], [C[i, 1], C[n + i, 1]], color=GRID, lw=0.5, zorder=1)
    for sc, mk in (("ar", "o"), ("az", "^")):
        for c in ("neg", "pos"):
            m = (script == sc) & (sent == c)
            if sc == "ar":
                ax.scatter(C[m, 0], C[m, 1], s=s, marker=mk, color=COL[c], edgecolor="white", lw=0.4, zorder=3)
            else:
                ax.scatter(C[m, 0], C[m, 1], s=s * 1.2, marker=mk, facecolor="white", edgecolor=COL[c], lw=0.9, zorder=3)


def legend(fig, y=0.005):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=INK, markeredgecolor="white", label="Arabic script (filled)"),
         Line2D([], [], ls="", marker="^", markerfacecolor="white", markeredgecolor=INK, label="Arabizi (open)"),
         Line2D([], [], ls="", marker="s", color=BLUE, label="negative"),
         Line2D([], [], ls="", marker="s", color=ORANGE, label="positive"),
         Line2D([], [], color=GRID, lw=1, label="same sentence")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y), fontsize=8.5)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


# 1. before/after with a FIXED basis per encoder
srcs = ["none", "human", "rule", "dict"]
fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.2))
for r, enc in enumerate(ENCN):
    allc = np.vstack([np.asarray(V["enc"][enc]["sources"][s]["pca3"])[:, :2] for s in srcs])
    lo, hi = allc.min(0), allc.max(0); pad = 0.06 * (hi - lo)
    for c, s in enumerate(srcs):
        ax = axes[r, c]
        scatter2(ax, np.asarray(V["enc"][enc]["sources"][s]["pca3"])[:, :2], s=11)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        clean(ax)
        m = G["enc"][enc][s]
        ax.text(0.03, 0.03, f"probe {m['script_probe_narabizi']:.0f}%  R@1 {m['retrieval_at1']:.0f}%", transform=ax.transAxes,
                fontsize=7.5, color=INK)
        if r == 0:
            ax.set_title(SRCN[s], fontsize=9.5)
        if c == 0:
            evr = V["enc"][enc]["pca_raw_evr"]
            ax.set_ylabel(f"{ENCN[enc]}\nPC2 ({100*evr[1]:.0f}%)", fontsize=9)
        if r == 2:
            ax.set_xlabel("PC1 of raw embeddings", fontsize=8)
legend(fig)
fig.subplots_adjust(0.07, 0.07, 0.995, 0.96, wspace=0.04, hspace=0.06)
fig.savefig(FIG / "fig_beforeafter_pca.pdf"); fig.savefig(FIG / "fig_beforeafter_pca.png", dpi=130)
plt.close(fig)


# 2. real 3-D PCA, fixed basis, mE5
def plot3(ax, C, title):
    C = np.asarray(C)
    for i in range(n):
        ax.plot([C[i, 0], C[n + i, 0]], [C[i, 1], C[n + i, 1]], [C[i, 2], C[n + i, 2]], color=GRID, lw=0.4)
    for sc, mk in (("ar", "o"), ("az", "^")):
        for c in ("neg", "pos"):
            m = (script == sc) & (sent == c)
            if sc == "ar":
                ax.scatter(C[m, 0], C[m, 1], C[m, 2], s=12, marker=mk, color=COL[c], depthshade=False, edgecolor="white", lw=0.3)
            else:
                ax.scatter(C[m, 0], C[m, 1], C[m, 2], s=15, marker=mk, facecolor="white", edgecolor=COL[c], depthshade=False, lw=0.8)
    ax.set_title(title, fontsize=9.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor("#e6e6e6")
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=20, azim=-55)


def fig3d(enc, path, srcs3):
    evr = V["enc"][enc]["pca_raw_evr"]
    nc = 2 if len(srcs3) == 4 else len(srcs3); nr = (len(srcs3) + nc - 1) // nc
    fig = plt.figure(figsize=(3.7 * nc, 3.75 * nr + 0.2))
    allc = np.vstack([np.asarray(V["enc"][enc]["sources"][s]["pca3"]) for s in srcs3])
    lo, hi = allc.min(0), allc.max(0)
    for j, s in enumerate(srcs3):
        ax = fig.add_subplot(nr, nc, j + 1, projection="3d")
        cap = V["enc"][enc]["sources"][s]["pca_fixed_var_captured"]
        plot3(ax, V["enc"][enc]["sources"][s]["pca3"], f"{ENCN[enc]}: {SRCN[s]}\n(axes hold {100*cap:.0f}% of this set's variance)")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel(f"PC1 ({100*evr[0]:.0f}%)", fontsize=7, labelpad=-6)
        ax.set_ylabel(f"PC2 ({100*evr[1]:.0f}%)", fontsize=7, labelpad=-6)
        ax.set_zlabel(f"PC3 ({100*evr[2]:.0f}%)", fontsize=7, labelpad=-7)
    legend(fig, y=-0.01)
    fig.subplots_adjust(0, 0.08 / nr, 0.98, 1 - 0.12 / nr, wspace=0.0, hspace=0.12)
    fig.savefig(FIG / f"{path}.pdf"); fig.savefig(FIG / f"{path}.png", dpi=130)
    plt.close(fig)




def fig3axes(enc, path, srcs3):
    nc = 2 if len(srcs3) == 4 else len(srcs3); nr = (len(srcs3) + nc - 1) // nc
    fig = plt.figure(figsize=(3.7 * nc, 3.75 * nr + 0.2))
    A0 = np.asarray(V["enc"][enc]["sources"]["none"]["axes3"]); sd = A0.std(0)
    allc = np.vstack([np.asarray(V["enc"][enc]["sources"][s]["axes3"]) / sd for s in srcs3])
    lo, hi = allc.min(0), allc.max(0)
    for j, s in enumerate(srcs3):
        ax = fig.add_subplot(nr, nc, j + 1, projection="3d")
        plot3(ax, np.asarray(V["enc"][enc]["sources"][s]["axes3"]) / sd, f"{ENCN[enc]}: {SRCN[s]}")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("script axis $u_1$", fontsize=7.5, labelpad=-6)
        ax.set_ylabel("sentiment axis $s$", fontsize=7.5, labelpad=-6)
        ax.set_zlabel("residual axis $r$", fontsize=7.5, labelpad=-7)
    legend(fig, y=-0.01)
    fig.subplots_adjust(0, 0.08 / nr, 0.98, 1 - 0.08 / nr, wspace=0.0, hspace=0.08)
    fig.savefig(FIG / f"{path}.pdf"); fig.savefig(FIG / f"{path}.png", dpi=130)
    plt.close(fig)


fig3axes("e5-small", "fig_3d_real", ["none", "human"])
for enc in ENCN:
    fig3axes(enc, f"figS3b_3d_axes_{enc}", ["none", "rule", "dict", "human"])
for enc in ENCN:
    fig3d(enc, f"figS3_pca3d_{enc}", ["none", "human", "rule", "dict"])

# 3. diagnostics dot plot
srcs6 = ["none", "rule", "dict", "neural", "human", "tarc"]
metrics = [("script_probe_narabizi", "script probe (%)\nNArabizi dev", (45, 101), 50),
           ("script_probe_tsac", "script probe (%)\nTSAC dev", (45, 101), 50),
           ("retrieval_at1", "pair retrieval@1 (%)\nNArabizi dev", (0, 65), None),
           ("heldout_residual_energy", "residual pair energy\nNArabizi dev", (0.2, 1.03), None)]
mk = {"e5-small": ("o", BLUE), "labse": ("s", ORANGE), "xlmr-base": ("D", AQUA)}
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
for j, (key, name, xl, chance) in enumerate(metrics):
    ax = axes[j]
    for k, enc in enumerate(ENCN):
        xs = [G["enc"][enc][s][key] for s in srcs6]
        ys = np.arange(len(srcs6)) + (k - 1) * 0.2
        ax.scatter(xs, ys, marker=mk[enc][0], color=mk[enc][1], s=20, label=ENCN[enc], zorder=3,
                   edgecolor="white", lw=0.4)
    if chance:
        ax.axvline(chance, color=INK, lw=0.6, ls=":")
    ax.set_xlim(*xl); ax.set_title(name, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5)
axes[0].set_yticks(range(len(srcs6))); axes[0].set_yticklabels([SRCN[s] for s in srcs6], fontsize=8)
axes[0].invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.55, -0.01))
fig.subplots_adjust(0.14, 0.2, 0.99, 0.88, wspace=0.12)
fig.savefig(FIG / "fig_diagnostics.pdf"); fig.savefig(FIG / "fig_diagnostics.png", dpi=130)
plt.close(fig)


# 4. t-SNE / UMAP (separately fitted per panel: axes not comparable; descriptive only)
def nonlinear(kind, path):
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.0))
    for r, enc in enumerate(ENCN):
        for c, s in enumerate(srcs):
            ax = axes[r, c]
            scatter2(ax, V["enc"][enc]["sources"][s][kind], s=9, lines=True)
            clean(ax)
            if r == 0:
                ax.set_title(SRCN[s], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(ENCN[enc], fontsize=9)
    legend(fig)
    fig.subplots_adjust(0.04, 0.06, 0.995, 0.96, wspace=0.04, hspace=0.05)
    fig.savefig(FIG / f"{path}.pdf"); fig.savefig(FIG / f"{path}.png", dpi=130)
    plt.close(fig)


nonlinear("tsne", "figS1_tsne")
nonlinear("umap", "figS2_umap")

# 5. TSAC (out of pair corpus), mE5, fixed basis fitted on raw TSAC sample
fig, axes = plt.subplots(1, 6, figsize=(7.4, 1.75))
T = {s: np.asarray(V["enc"]["e5-small"]["sources"][s]["tsac_pca3"])[:, :2] for s in srcs6}
allc = np.vstack(list(T.values())); lo, hi = allc.min(0), allc.max(0)
for j, s in enumerate(srcs6):
    ax = axes[j]; C = T[s]
    ax.scatter(C[:150, 0], C[:150, 1], s=6, marker="o", color=INK, alpha=0.8, lw=0)
    ax.scatter(C[150:, 0], C[150:, 1], s=7, marker="^", facecolor="white", edgecolor=ORANGE, lw=0.6)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); clean(ax)
    ax.set_title(f"{SRCN[s]}\nprobe {G['enc']['e5-small'][s]['script_probe_tsac']:.0f}%", fontsize=7.5)
fig.subplots_adjust(0.005, 0.02, 0.995, 0.75, wspace=0.05)
fig.savefig(FIG / "figS5_tsac_pca.pdf"); fig.savefig(FIG / "figS5_tsac_pca.png", dpi=130)
print("figures written")

# 6. interpretable fixed axes: u1 = leading human pair-difference direction (oriented Arabic > Arabizi),
#    s = Arabic-script train sentiment direction orthogonalised against u1. Same axes in every panel of a row.
srcs5 = ["none", "rule", "dict", "neural", "human"]
axstats = {}
fig, axes = plt.subplots(3, 5, figsize=(7.4, 4.9))
for r, enc in enumerate(ENCN):
    A0 = np.asarray(V["enc"][enc]["sources"]["none"]["axes2"])
    sd = A0.std(0)
    allc = np.vstack([np.asarray(V["enc"][enc]["sources"][s]["axes2"]) for s in srcs5]) / sd
    lo, hi = allc.min(0), allc.max(0); pad = 0.05 * (hi - lo)
    axstats[enc] = {}
    for c, s in enumerate(srcs5):
        Cc = np.asarray(V["enc"][enc]["sources"][s]["axes2"]) / sd
        gap = Cc[:n, 0].mean() - Cc[n:, 0].mean()
        sep_az = Cc[n:, 1][lab == "pos"].mean() - Cc[n:, 1][lab == "neg"].mean()
        sep_ar = Cc[:n, 1][lab == "pos"].mean() - Cc[:n, 1][lab == "neg"].mean()
        axstats[enc][s] = {"script_gap_u1_rawSD": float(gap), "sent_sep_arabizi_rawSD": float(sep_az),
                           "sent_sep_arabic_rawSD": float(sep_ar)}
        ax = axes[r, c]
        scatter2(ax, Cc, s=9)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1]); clean(ax)
        ax.text(0.03, 0.03, f"gap {gap:.1f}  sep$_{{az}}$ {sep_az:.1f}", transform=ax.transAxes, fontsize=7, color=INK)
        if r == 0:
            ax.set_title(SRCN[s], fontsize=9)
        if c == 0:
            ax.set_ylabel(f"{ENCN[enc]}\nsentiment axis $s$", fontsize=8.5)
        if r == 2:
            ax.set_xlabel("script axis $u_1$", fontsize=8)
legend(fig, y=0.0)
fig.subplots_adjust(0.075, 0.1, 0.995, 0.95, wspace=0.04, hspace=0.06)
fig.savefig(FIG / "fig_axes2.pdf"); fig.savefig(FIG / "fig_axes2.png", dpi=130)
(REPORTS_V3 / "axes_stats.json").write_text(json.dumps(axstats, indent=1))
print(json.dumps(axstats, indent=0)[:1500])
