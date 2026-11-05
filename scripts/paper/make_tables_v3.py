"""V3 tables, figure and in-text macros from reports/v3/*.json (+ saved test predictions for the error table).
Run after v3_prediction_analysis.py, (v3_geometry.py) and make_figures_v3.py:
    python scripts/paper/make_tables_v3.py
Writes $FORMAL_OUT/tables/v3_*.tex, $FORMAL_OUT/tables/v3_numbers.tex, $FORMAL_OUT/figures/fig_confusion.pdf.
Inputs regenerated under $FORMAL_OUT are preferred over the archived reports/v3 copies.
The example-sentence tables (v3_error_examples, v3_phenomena) need the rebuilt corpus and are skipped without it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formal_si.confirm import load
from formal_si.text.normalize import deromanize, normalize_arabic
from formal_si.translit import cer, cross_fitted, resources

from _paths import FIGURES, HAVE_CORPUS, HAVE_NEURAL_MAPS, TABLES, prefer_out

T = TABLES; F = FIGURES
PA = json.load(open(prefer_out("reports/v3/predictions_analysis.json")))
G = json.load(open(prefer_out("reports/v3/geometry_v3.json")))
AX = json.load(open(prefer_out("reports/v3/axes_stats.json")))
ENCS = ["e5-small", "labse", "xlmr-base"]
EN = {"e5-small": "mE5", "labse": "LaBSE", "xlmr-base": "XLM-R"}
TASKS = ["narabizi_sent", "tsac_sent", "myc_sent", "narabizi_topic"]
TN = {"narabizi_sent": "NArabizi sent.", "tsac_sent": "TSAC sent.", "myc_sent": "MYC sent.", "narabizi_topic": "NArabizi topic"}
MAC = {}


def mac(name, val, fmt="{:.1f}"):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MAC[name] = fmt.format(val) if not isinstance(val, str) else val


def tex(s):
    return s.replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_").replace("$", "\\$").replace("~", "-").replace("^", "")


def latin_runs(s):
    return re.sub(r"([^\u0600-\u06FF\s]+(?:\s+[^\u0600-\u06FF\s]+)*)", lambda m: "\\foreignlanguage{english}{" + m.group(1) + "}", s)


# ---------------- geometry table (dev only)
SRC = ["none", "rule", "dict", "neural", "human", "tarc"]
SN = {"none": "none", "rule": "rule", "dict": "dictionary", "neural": "neural$^{\\ast}$", "human": "human",
      "tarc": "TArC$^{\\ast}$"}
rows = []
for enc in ENCS:
    for i, s in enumerate(SRC):
        m = G["enc"][enc][s]; c = m["clf"]
        ov = "--" if s in ("none", "human") else f"{G['enc'][enc]['overlap_with_human'][s]['mean_cos2']:.2f}"
        ov = "1" if s == "human" else ov
        se = "--" if s == "none" else f"{m['sent_dir_energy_removed']:.2f}"
        gap = AX[enc][s]["script_gap_u1_rawSD"] if s in AX[enc] else float("nan")
        gaps = "--" if s == "tarc" else f"{abs(gap):.2f}"
        rows.append(f"{EN[enc] if i == 0 else ''} & {SN[s]} & {m['script_probe_narabizi']:.1f} & {m['script_probe_tsac']:.1f} & "
                    f"{m['retrieval_at1']:.1f} & {m['heldout_residual_energy']:.2f} & {ov} & {se} & {gaps} & "
                    f"{c['agreement']:.1f} & {c['tightness']:.3f} \\\\")
    rows.append("\\addlinespace")
(T / "v3_geometry.tex").write_text(
    "\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n"
    " & & \\multicolumn{2}{c}{script probe (\\%)} & & & & & & \\multicolumn{2}{c}{sentiment pairs}\\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(l){10-11}\n"
    "Enc. & pairs & NArabizi & TSAC & R@1 & $\\rho_{\\mathrm{res}}$ & $\\overline{\\cos^2\\theta}$ & $E^{s}_{16}$ & gap$_{u_1}$ & agree & $\\tau$\\\\\n\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
for enc in ENCS:
    k = {"e5-small": "E", "labse": "L", "xlmr-base": "X"}[enc]
    for s in SRC:
        m = G["enc"][enc][s]
        sc = s.capitalize()
        mac(f"probe{k}{sc}", m["script_probe_narabizi"]); mac(f"probeT{k}{sc}", m["script_probe_tsac"])
        mac(f"ret{k}{sc}", m["retrieval_at1"]); mac(f"agr{k}{sc}", m["clf"]["agreement"])
        mac(f"res{k}{sc}", m["heldout_residual_energy"], "{:.2f}")
        if s in AX[enc]:
            mac(f"sepAz{k}{sc}", AX[enc][s]["sent_sep_arabizi_rawSD"], "{:.2f}")
            mac(f"sepAr{k}{sc}", AX[enc][s]["sent_sep_arabic_rawSD"], "{:.2f}")
            mac(f"gap{k}{sc}", abs(AX[enc][s]["script_gap_u1_rawSD"]), "{:.2f}")
    for s in ("rule", "dict", "neural", "tarc"):
        mac(f"ov{k}{s.capitalize()}", G["enc"][enc]["overlap_with_human"][s]["mean_cos2"], "{:.2f}")
allv = lambda f: [f(enc, s) for enc in ENCS for s in SRC]
tight = allv(lambda e, s: G["enc"][e][s]["clf"]["tightness"])
mac("tightMin", min(tight), "{:.3f}"); mac("tightMax", max(tight), "{:.3f}")
mac("certMax", max(allv(lambda e, s: G["enc"][e][s]["clf"]["certified"])), "{:.0f}")
mac("idDev", max(G["enc"][e][s].get("identity_max_abs_dev", 0) for e in ENCS for s in SRC), "{:.1e}")
mac("nDevPairs", G["meta"]["n_dev_pairs"], "{:d}"); mac("nDevSentPairs", G["meta"]["n_dev_sent_pairs"], "{:d}")
mac("remEnergyEMin", min(G["enc"]["e5-small"][s]["mean_removed_energy"] for s in SRC[1:]), "{:.3f}")
mac("remEnergyLMin", min(G["enc"]["labse"][s]["mean_removed_energy"] for s in SRC[1:]), "{:.2f}")
mac("remEnergyLMax", max(G["enc"]["labse"][s]["mean_removed_energy"] for s in SRC[1:]), "{:.2f}")
mac("remEnergyXMax", max(G["enc"]["xlmr-base"][s]["mean_removed_energy"] for s in SRC[1:]), "{:.3f}")
V = json.load(open(prefer_out("reports/v3/viz_coords.json")))
mac("pcaRawEvrE", 100 * sum(V["enc"]["e5-small"]["pca_raw_evr"]), "{:.0f}")
mac("pcaCapHumanE", 100 * V["enc"]["e5-small"]["sources"]["human"]["pca_fixed_var_captured"], "{:.0f}")
mac("pcaCapNoneE", 100 * V["enc"]["e5-small"]["sources"]["none"]["pca_fixed_var_captured"], "{:.0f}")

# ---------------- per-class F1 (cross-script, mean over encoders)
SYS = [("S0_raw", "S0 none"), ("S1_human", "S1 human pairs"), ("S3_dict_pairs", "S3 dict.\\ pairs"),
       ("S4c_derom_neural", "S4c neural input$^{\\ast}$")]
rows = []
for t in TASKS:
    labels = sorted(PA["per_class"][f"{t}|e5-small|S0_raw|cross_script"])
    pad = " & " * (5 - len(labels))
    sup = [PA["per_class"][f"{t}|e5-small|S0_raw|cross_script"][c]["support"] // 5 for c in labels]
    # header row: class names and Arabizi test support per class
    rows.append(f"\\textbf{{{TN[t]}}} & \\textit{{class (support)}} & "
                + " & ".join(f"\\textit{{{c.lower()}}} ({n})" for c, n in zip(labels, sup)) + pad + "\\\\")
    for j, (s, sname) in enumerate(SYS):
        vals = [np.mean([PA["per_class"][f"{t}|{e}|{s}|cross_script"][c]["f1"] for e in ENCS]) for c in labels]
        rows.append(f" & {sname} & " + " & ".join(f"{v:.1f}" for v in vals) + pad + "\\\\")
    rows.append("\\addlinespace")
(T / "v3_perclass.tex").write_text("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nTask & system & \\multicolumn{5}{c}{per-class F1 (\\%), cross-script test, mean over three encoders}\\\\\n\\midrule\n"
                                   + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
# topic: per-class F1 of S0 vs S1 for macros
topic_lab = sorted(PA["per_class"]["narabizi_topic|e5-small|S0_raw|cross_script"])
for s, k in (("S0_raw", "None"), ("S1_human", "Human"), ("S4c_derom_neural", "Neural")):
    for c in topic_lab:
        mac(f"topic{k}{c.capitalize()}", np.mean([PA["per_class"][f"narabizi_topic|{e}|{s}|cross_script"][c]["f1"] for e in ENCS]))
for t, kk in (("narabizi_sent", "Nar"), ("tsac_sent", "Tsac"), ("myc_sent", "Myc")):
    for s, k in (("S0_raw", "None"), ("S1_human", "Human")):
        for c in ("neg", "pos"):
            mac(f"rec{kk}{k}{c.capitalize()}", np.mean([PA["per_class"][f"{t}|{e}|{s}|cross_script"][c]["recall"] for e in ENCS]))

# ---------------- confusion figure (NArabizi topic, cross-script, mE5, rows normalised)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (s, name) in zip(axes, [("S0_raw", "S0: no projection"), ("S1_human", "S1: human-pair projection"),
                                ("S4c_derom_neural", "S4c: neural input")]):
    cm = np.zeros((5, 5))
    for e in ENCS:
        d = PA["confusion"][f"narabizi_topic|{e}|{s}|cross_script"]
        cm += np.array(d["matrix"])
    cm = cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{100*cm[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.55 else "#222222")
    lab = [l.capitalize()[:5] for l in d["labels"]]
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7, rotation=40)
    ax.set_yticks(range(5)); ax.set_yticklabels(lab if ax is axes[0] else [], fontsize=7)
    ax.set_title(name, fontsize=8.5); ax.set_xlabel("predicted", fontsize=7.5)
axes[0].set_ylabel("gold", fontsize=7.5)
plt.rcParams.update({"font.family": "serif"})
fig.subplots_adjust(0.08, 0.2, 0.99, 0.9, wspace=0.08)
fig.savefig(F / "fig_confusion.pdf"); fig.savefig(F / "fig_confusion.png", dpi=130)

# ---------------- error categories (majority vote over 5 resamples), with exact McNemar on S0 vs S1
def mcnemar(b, c):
    n = b + c
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return min(1.0, p)


rows = []
for t in TASKS:
    for j, e in enumerate(ENCS):
        d = PA["error_categories"][f"{t}|{e}"]
        b, c = d.get("projection_corrects", 0), d.get("projection_degrades", 0)
        p = mcnemar(b, c)
        rows.append(f"{TN[t] if j == 0 else ''} & {EN[e]} & {d['n']} & {b} & {c} & {('$<$0.001' if p < 0.001 else f'{p:.3f}')} & {d.get('input_translit_wins', 0)} & "
                    f"{d.get('projection_wins_over_input', 0)} & {d.get('all_fail', 0)} \\\\")
        pass
    rows.append("\\addlinespace")
(T / "v3_errors.tex").write_text("\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
                                 " & & & \\multicolumn{3}{c}{S1 vs.\\ S0} & \\multicolumn{2}{c}{S4c vs.\\ S1} & \\\\\n"
                                 "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
                                 "Task & Enc. & $n$ & corrected & degraded & $p_{\\mathrm{McN}}$ & S4c only & S1 only & all fail\\\\\n\\midrule\n"
                                 + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
tot = lambda key, t: sum(PA["error_categories"][f"{t}|{e}"].get(key, 0) for e in ENCS)
mac("narCorr", tot("projection_corrects", "narabizi_sent"), "{:d}"); mac("narDegr", tot("projection_degrades", "narabizi_sent"), "{:d}")
mac("tsacCorr", tot("projection_corrects", "tsac_sent"), "{:d}"); mac("tsacDegr", tot("projection_degrades", "tsac_sent"), "{:d}")
mac("mycCorr", tot("projection_corrects", "myc_sent"), "{:d}"); mac("mycDegr", tot("projection_degrades", "myc_sent"), "{:d}")
mac("topCorr", tot("projection_corrects", "narabizi_topic"), "{:d}"); mac("topDegr", tot("projection_degrades", "narabizi_topic"), "{:d}")
mac("topAllFail", tot("all_fail", "narabizi_topic"), "{:d}"); mac("topN", 3 * 144, "{:d}")
mac("tsacInWins", tot("input_translit_wins", "tsac_sent"), "{:d}"); mac("tsacProjWins", tot("projection_wins_over_input", "tsac_sent"), "{:d}")

# ---------------- error examples: NArabizi sentiment test, mE5, first item per category in md5(group) order
# Needs the rebuilt corpus and the neural-transliteration maps (not distributed; see docs/DATASETS.md).
if not (HAVE_CORPUS and HAVE_NEURAL_MAPS):
    print("corpus or neural maps not found: skipping v3_error_examples and v3_phenomena")
    (T / "v3_numbers.tex").write_text("% generated by scripts/make_tables_v3.py; do not edit\n" +
                                      "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())) + "\n")
    print(len(MAC), "v3 macros")
    raise SystemExit(0)
ex = load(include_test=True)
gold = json.load(open("runs/confirm/test/gold.json"))["narabizi_sent"]["cross_script"]
test_rows = [e for e in ex if e.source == "narabizi" and e.task == "sentiment" and e.label and e.split == "test" and e.form == "arabizi"]
assert [str(e.label) for e in test_rows] == gold["labels"]
nmap = json.loads(Path("cache_neural_translit/full_input_map.json").read_text())
E = PA["examples"]["narabizi_sent|e5-small"]; votes = E["votes"]
order = sorted(range(len(test_rows)), key=lambda i: hashlib.md5(test_rows[i].group.encode()).hexdigest())
cats = [("projection_corrects", "projection corrects"), ("projection_degrades", "projection degrades"),
        ("input_translit_wins", "input transliteration wins"), ("all_fail", "all three fail")]
rows, js = [], []
for key, name in cats:
    pick = [i for i in order if i in set(E[key]) and 3 <= len(test_rows[i].text.split()) <= 9 and "http" not in test_rows[i].text][:1]
    for i in pick:
        e = test_rows[i]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{name}}}: \\texttt{{{tex(e.text)}}}}}\\\\")
        rows.append(f" & \\ar{{{latin_runs(tex(nmap.get(e.text, '')))}}} & {e.label} & {votes['S0'][i]} / {votes['S1'][i]} / {votes['S4c'][i]} & \\\\\\addlinespace")
        js.append({"group": e.group, "category": key, "arabizi": e.text, "neural_input": nmap.get(e.text), "gold": e.label,
                   "S0": votes["S0"][i], "S1": votes["S1"][i], "S4c": votes["S4c"][i]})
(T / "v3_error_examples.tex").write_text("\\begin{tabular}{@{}lllll@{}}\n\\toprule\n & neural transliteration of the input (S4c) & gold & S0 / S1 / S4c & \\\\\n\\midrule\n"
                                         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
(T / "v3_error_examples.json").write_text(json.dumps(js, ensure_ascii=False, indent=1))

# ---------------- linguistic phenomena table (TRAIN sentences, mechanical selection)
trdev = [e for e in ex if e.split != "test"]
_, nar = resources(trdev)
cf_dict = cross_fitted(trdev)
cf_neur = json.loads(Path("cache_neural_translit/cross_fitted_pairs.json").read_text())
cs = {}
for e in trdev:
    if e.source == "narabizi" and e.form == "codeswitch":
        cs.setdefault(e.group, e.text)
ordr = sorted(nar, key=lambda g: hashlib.md5(g.encode()).hexdigest())
ok = lambda g: 4 <= len(nar[g]["arabizi"].split()) <= 8 and "http" not in nar[g]["arabizi"] and "@" not in nar[g]["arabizi"]
digit = [g for g in ordr if ok(g) and re.search(r"[a-z][2-9]|[2-9][a-z]", nar[g]["arabizi"].lower())]
latin = lambda g: [w for w in cs.get(g, "").split() if re.fullmatch(r"[A-Za-zÀ-ÿ'’]{3,}", w)]
french = [g for g in ordr if ok(g) and len(latin(g)) >= 2 and g not in digit[:2]]
picks = [(g, "digit letters") for g in digit[:2]] + [(g, "French tokens") for g in french[:2]]
rows, js = [], []
for g, why in picks:
    az, ar = nar[g]["arabizi"], nar[g]["arabic"]; ref = normalize_arabic(ar)
    outs = {"rule": normalize_arabic(deromanize(az)), "dictionary": cf_dict[g], "neural": cf_neur[g]}
    digs = sorted(set(re.findall(r"[2-9]", az)))
    feat = f"digits {','.join(digs)}" if digs else ""
    lat = latin(g)
    feat += ("; " if feat else "") + (f"{len(lat)} Latin-script tokens in the code-switched layer" if lat else "no Latin token in the code-switched layer")
    rows.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{Arabizi}} \\texttt{{{tex(az)}}}\\quad\\textit{{({tex(feat)})}}}}\\\\")
    if g in cs:
        rows.append(f" & code-switched layer & \\ar{{{latin_runs(tex(cs[g]))}}} \\\\")
    rows.append(f" & human Arabic script & \\ar{{{latin_runs(tex(ar))}}} \\\\")
    for k, v in outs.items():
        rows.append(f" & {k} (CER {cer(v, ref):.2f}) & \\ar{{{latin_runs(tex(v))}}} \\\\")
    rows.append("\\addlinespace")
    js.append({"group": g, "selected_for": why, "arabizi": az, "codeswitch": cs.get(g), "human_arabic": ar, **outs,
               **{f"cer_{k}": cer(v, ref) for k, v in outs.items()}})
(T / "v3_phenomena.tex").write_text("\\begin{tabular}{@{}lll@{}}\n\\toprule\n & layer or source & text\\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
                                    "\n\\bottomrule\n\\end{tabular}\n")
(T / "v3_phenomena.json").write_text(json.dumps(js, ensure_ascii=False, indent=1))

(T / "v3_numbers.tex").write_text("% generated by scripts/make_tables_v3.py; do not edit\n" +
                                  "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())) + "\n")
print(len(MAC), "v3 macros; examples", [j["group"] for j in js])
