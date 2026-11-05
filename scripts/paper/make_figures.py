"""Data figures for the revised manuscript, generated only from saved run/report files.
Usage:   python scripts/paper/make_figures.py   (writes to $FORMAL_OUT/figures)
Schematic figures (A pipeline, C geometry) are TikZ in the LaTeX source; B is a generated table.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from _paths import FIGURES

PKG = Path(".")
OUT = FIGURES
# Validated categorical slots 1-3 (all-pairs pass); a 4th source uses neutral dark ink + dashes + direct labels.
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SRC = {"human": (BLUE, "o", "-", "human (NArabizi)"), "dict": (ORANGE, "s", "-", "dictionary"),
       "rule": (AQUA, "^", "-", "rule"), "tarc_human": (INK2, "D", "--", "human (TArC, Tunisian)")}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "pdf.fonttype": 42})
TASKS = ["narabizi_sent", "tsac_sent", "myc_sent", "narabizi_topic"]
TN = {"narabizi_sent": "NArabizi sent.", "tsac_sent": "TSAC sent.", "myc_sent": "MYC sent.", "narabizi_topic": "NArabizi topic"}
ENCS = ["e5-small", "labse", "xlmr-base"]
EN = {"e5-small": "mE5-small", "labse": "LaBSE (CLS)", "xlmr-base": "XLM-R", "labse-st": "LaBSE (ST head)"}
G = json.load(open(PKG / "reports/geometry/geometry.json"))
ST = json.load(open(PKG / "runs/confirm/test/stats.json"))
DEP = json.load(open(PKG / "reports/depstats/test.json"))
prov = {}


def save(fig, name, sources):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    prov[name] = sources


# ---- E: eigenvalue spectra ----
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2), sharey=True)
for ax, e in zip(axes, ENCS):
    for s in ("human", "dict", "rule", "tarc_human"):
        col, mk, ls, lab = SRC[s]
        v = np.array(G[e]["spectra"][s]["eig_normalised"][:64])
        ax.plot(np.arange(1, 65), np.cumsum(v), color=col, ls=ls, lw=1.6, label=lab)
    ax.axvline(16, color=GRAY, lw=0.8, ls=":")
    ax.set_title(EN[e], fontsize=8)
    ax.set_xlabel("number of eigenvectors $k$")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"])
axes[0].set_ylabel("cumulative share of tr$(S)$")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6.5, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "fig_E_spectra", ["reports/geometry/geometry.json:spectra"])

# ---- F: subspace similarity (principal angles) ----
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
pairs = [("dict|human", "human vs dictionary", ORANGE, "-", "dict."), ("human|rule", "human vs rule", AQUA, "-", "rule"),
         ("human|tarc_human", "NArabizi vs TArC human", INK2, "--", "TArC")]
for ax, e in zip(axes, ENCS):
    for k, lab, col, ls, sh in pairs:
        ang = G[e]["subspaces"][k]["angles_deg"]
        ax.plot(range(1, 17), ang, color=col, ls=ls, lw=1.6, marker="o", ms=2.5, label=lab)
    ov = ", ".join(f"{sh} {G[e]['subspaces'][k]['overlap']:.2f}" for k, _, _, _, sh in pairs)
    ax.axhline(90, color=GRAY, lw=0.6)
    ax.set_title(f"{EN[e]}\noverlap: {ov}\n(random {G[e]['random_overlap_expected']:.3f})", fontsize=6.5)
    ax.set_xlabel("principal angle index")
    ax.set_ylim(0, 92)
axes[0].set_ylabel("principal angle (degrees)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6.5, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "fig_F_angles", ["reports/geometry/geometry.json:subspaces"])

# ---- D: PCA small multiples (NArabizi dev pairs) ----
fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.2))
for i, e in enumerate(ENCS):
    P = json.load(open(PKG / f"reports/geometry/pca_{e}.json"))
    H = G[e]["heldout_energy_removed"]["narabizi_dev"]
    for j, (st_, title) in enumerate([("before", "no removal"), ("after_human", "after $P_{16}$ (human pairs)"),
                                      ("after_rule", "after $P_{16}$ (rule pairs)")]):
        ax = axes[i, j]
        A, B = np.array(P[st_]["ar"]), np.array(P[st_]["az"])
        for a, b in zip(A, B):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#d8d7d2", lw=0.4, zorder=1)
        ax.scatter(A[:, 0], A[:, 1], s=8, color=BLUE, edgecolor="white", lw=0.3, label="Arabic script", zorder=2)
        ax.scatter(B[:, 0], B[:, 1], s=8, color=ORANGE, marker="s", edgecolor="white", lw=0.3, label="Arabizi", zorder=2)
        ex = P[st_]["explained"]
        src = {"before": "none", "after_human": "human", "after_rule": "rule"}[st_]
        diag = f"probe acc. {G[e]['script_probe'][src]:.2f}"
        if st_ != "before":
            diag += f", energy removed {H[src]['energy_removed']:.2f}"
        ax.set_title(f"{EN[e]}: {title}\n{diag}\nPC1+PC2 = {100*(ex[0]+ex[1]):.0f}% of variance", fontsize=6.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7, loc="upper center", ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, "fig_D_pca", ["reports/geometry/pca_*.json", "reports/geometry/geometry.json:heldout_energy_removed"])

# ---- G: forest plots, 12 cells + dependency-aware aggregates ----
Hs = ST["hypotheses"]
cells = [f"{t}|{e}" for t in TASKS for e in ENCS]
fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.3), sharey=True)
for ax, (h, title, col) in zip(axes, [("H1 S1>S0", "H1: human pairs $-$ no removal", BLUE),
                                     ("H2 S1>S2", "H2: human $-$ rule pairs", AQUA),
                                     ("H3 S1>S3", "H3: human $-$ dictionary pairs", ORANGE)]):
    y = 0
    ticks, labs = [], []
    for c in cells:
        v = Hs[h]["cells"][c]
        ax.plot([v["lo"], v["hi"]], [y, y], color=col, lw=1.4)
        ax.plot(v["delta"], y, "o", color=col, ms=3.8, mec="white", mew=0.6)
        if v["p_holm"] < 0.05:
            ax.text(v["hi"] + 0.4, y, "*", va="center", fontsize=7, color=INK)
        ticks.append(y); labs.append(f"{TN[c.split('|')[0]]} / {EN[c.split('|')[1]].split(' ')[0]}")
        y += 1
    y += 0.6
    D = DEP[h]
    for key, lab in [("cell_mean_joint", "12-cell mean (joint bootstrap)"), ("corpus_weighted_mean", "corpus-weighted mean"),
                     ("out_of_narabizi_mean", "TSAC+MYC mean")]:
        v = D[key]
        ax.plot([v["lo"], v["hi"]], [y, y], color=INK, lw=1.6)
        ax.plot(v["observed"], y, "D", color=INK, ms=4, mec="white", mew=0.6)
        ticks.append(y); labs.append(lab)
        y += 1
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("$\\Delta$ cross-script F1")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ticks); axes[0].set_yticklabels(labs, fontsize=6)
axes[0].invert_yaxis()
fig.tight_layout()
save(fig, "fig_G_forest", ["runs/confirm/test/stats.json:hypotheses", "reports/depstats/test.json"])

# ---- H: learning curve with draw-level uncertainty (dev) ----
rows = [json.loads(l) for l in open(PKG / "runs/confirm/curve_dev/metrics.jsonl")]
raw_dev = json.load(open(PKG / "runs/confirm/dev/stats.json"))["summary"]
base = np.mean([raw_dev[f"{t}|{e}|S0_raw"]["cross_script"] for t in TASKS for e in ENCS])
by = defaultdict(list)
for r in rows:
    by[(r["pairs"], r["n_pairs"], r["draw"])].append(100 * r["f1_cross_script"])
ns = [50, 100, 250, 500, 998]
fig, ax = plt.subplots(figsize=(3.1, 2.2))
curve_tab = {}
for src in ("human", "dict", "rule"):
    col, mk, _, lab = SRC[src]
    means, lo, hi = [], [], []
    for n in ns:
        dm = [np.mean(by[(src, n, d)]) for d in range(3) if (src, n, d) in by]
        means.append(np.mean(dm)); lo.append(min(dm)); hi.append(max(dm))
    curve_tab[src] = {"n": ns, "mean": means, "min_draw": lo, "max_draw": hi}
    ax.fill_between(ns, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(ns, means, color=col, lw=1.8, marker=mk, ms=3.5)
    ax.annotate(lab.split(" (")[0] + " pairs", (ns[-1], means[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=6.5, color=INK)
ax.axhline(base, color=GRAY, lw=1, ls="--")
ax.annotate("no removal", (50, base), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=6.5)
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
ax.set_xlim(40, 2900)
ax.set_xlabel("pairs used to estimate $S$ (log scale)")
ax.set_ylabel("cross-script F1, dev\n(mean of 12 cells)")
fig.tight_layout()
save(fig, "fig_H_curve", ["runs/confirm/curve_dev/metrics.jsonl", "runs/confirm/dev/stats.json:summary"])
json.dump({"curve": curve_tab, "no_removal_dev": base}, open(OUT / "fig_H_curve_values.json", "w"), indent=1)

# ---- I: robustness-preservation trade-off (test) ----
S = ST["summary"]
systems = [("S1_human", "human pairs", BLUE, "o"), ("S3_dict_pairs", "dict. pairs", BLUE, "s"),
           ("S2_rule_pairs", "rule pairs", BLUE, "^"), ("S4b_derom_dict", "dict. input", ORANGE, "s"),
           ("S4_derom_rule", "rule input", ORANGE, "^")]
fig, ax = plt.subplots(figsize=(3.1, 2.4))
for s, lab, col, mk in systems:
    dx = [S[f"{t}|{e}|{s}"]["in_script"] - S[f"{t}|{e}|S0_raw"]["in_script"] for t in TASKS for e in ENCS]
    dy = [S[f"{t}|{e}|{s}"]["cross_script"] - S[f"{t}|{e}|S0_raw"]["cross_script"] for t in TASKS for e in ENCS]
    ax.scatter(dx, dy, s=7, color=col, marker=mk, alpha=0.35, lw=0)
    ax.scatter(np.mean(dx), np.mean(dy), s=34, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3)
    off = {"S1_human": (-38, 4), "S3_dict_pairs": (6, -2), "S2_rule_pairs": (6, -3), "S4b_derom_dict": (6, 3),
           "S4_derom_rule": (6, -3)}[s]
    ax.annotate(lab, (np.mean(dx), np.mean(dy)), xytext=off, textcoords="offset points", fontsize=6.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
ax.axhline(0, color=INK2, lw=0.7); ax.axvline(0, color=INK2, lw=0.7)
ax.set_xlabel("$\\Delta$ in-script F1 vs. no removal")
ax.set_ylabel("$\\Delta$ cross-script F1 vs. no removal")
ax.set_title("blue: remove subspace (raw input); orange: transliterate input", fontsize=6, color=INK2)
fig.tight_layout()
save(fig, "fig_I_tradeoff", ["runs/confirm/test/stats.json:summary"])

# ---- J: failure analysis of the selective objective ----
PT = json.load(open(PKG / "reports/pilot/pilot_analysis.json"))["tests"]
pm = {"narabizi_sent": "pilot_narabizi", "tsac_sent": "pilot_tsac", "myc_sent": "pilot_myc", "narabizi_topic": "pilot_narabizi_topic"}
xs, ys, cost, labsJ, ang = [], [], [], [], []
for t in TASKS:
    for e in ENCS:
        xs.append(G[e]["task_overlap"][t]["arabic"]["E_16_human"])
        ang.append(G[e]["task_overlap"][t]["arabic"]["angles_task_to_human_deg"][0])
        ys.append(PT[f"loco|{pm[t]}|{e}|selective (best lam0>0) vs invariance-only"]["delta"])
        cost.append(S[f"{t}|{e}|S1_human"]["in_script"] - S[f"{t}|{e}|S0_raw"]["in_script"])
        labsJ.append((t, e))
rho1, p1 = spearmanr(xs, ys)
rho2, p2 = spearmanr(xs, cost)
fig, axes = plt.subplots(2, 1, figsize=(3.1, 4.6))
mk = {"e5-small": "o", "labse": "s", "xlmr-base": "^"}
for ax, yy, yl, (rho, p) in [(axes[0], ys, "selective $-$ invariance-only\n$\\Delta$ cross-script F1 (pilot dev)", (rho1, p1)),
                             (axes[1], cost, "$\\Delta$ in-script F1, human pairs\nvs. no removal (test)", (rho2, p2))]:
    for (t, e), x, y in zip(labsJ, xs, yy):
        ax.scatter(x, y, color=BLUE, marker=mk[e], s=18, edgecolor="white", lw=0.5)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("$E_{16}$: share of Arabic-script task scatter\nin the human-pair subspace")
    ax.set_ylabel(yl, fontsize=7)
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f} (p = {p:.2f}, n = 12 cells)", fontsize=7, color=INK2)
for e in ENCS:
    axes[0].scatter([], [], color=BLUE, marker=mk[e], s=18, label=EN[e])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6.5, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "fig_J_overlap", ["reports/geometry/geometry.json:task_overlap", "reports/pilot/pilot_analysis.json:tests",
                            "runs/confirm/test/stats.json:summary"])
json.dump({"E16_vs_selective": {"rho": rho1, "p": p1}, "E16_vs_inscript_cost": {"rho": rho2, "p": p2},
           "E16_range": [min(xs), max(xs)], "first_angle_deg_range": [min(ang), max(ang)]},
          open(OUT / "fig_J_values.json", "w"), indent=1)
json.dump(prov, open(OUT / "PROVENANCE.json", "w"), indent=1)
print("figures:", ", ".join(prov))
