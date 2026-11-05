"""Every LaTeX table and every number macro of the revised paper, generated from saved files only.
Usage (from anywhere):  python scripts/paper/make_tables.py
Writes $FORMAL_OUT/tables/*.tex and $FORMAL_OUT/tables/numbers.tex (\\newcommand macros used in the text).
Without the locally rebuilt corpus (data/examples.jsonl) the data-statistics table and its macros are skipped.
v2 (post-confirmatory) tables are written only if runs/confirm2/<split>/stats.json exists.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from _paths import HAVE_CORPUS, TABLES, prefer_out

PKG = Path(".")
OUT = TABLES
TASKS = ["narabizi_sent", "tsac_sent", "myc_sent", "narabizi_topic"]
TN = {"narabizi_sent": "NArabizi sent.", "tsac_sent": "TSAC sent.", "myc_sent": "MYC sent.", "narabizi_topic": "NArabizi topic"}
ENCS = ["e5-small", "labse", "xlmr-base"]
EN = {"e5-small": "mE5-small", "labse": "LaBSE", "xlmr-base": "XLM-R", "labse-st": "LaBSE-ST", "none": "char n-gram"}
ST = json.load(open(PKG / "runs/confirm/test/stats.json"))
SD = json.load(open(PKG / "runs/confirm/dev/stats.json"))
DEP = json.load(open(PKG / "reports/depstats/test.json"))
G = json.load(open(PKG / "reports/geometry/geometry.json"))
TQ = json.load(open(PKG / "reports/confirm/translit_quality.json"))
NQ = json.load(open(PKG / "artifacts/neural_translit/quality.json"))
M = {}  # number macros


def mac(name, val, fmt="{:.2f}"):
    M[name] = fmt.format(val) if not isinstance(val, str) else val


def sgn(v, d=2):
    return f"{v:+.{d}f}".replace("-", "$-$")


def ci(lo, hi, d=2):
    return f"[{sgn(lo, d)}, {sgn(hi, d)}]"


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------------- data statistics ----------------
def data_stats():
    from formal_si.confirm import load, TASKS as CT
    ex = load(include_test=True)
    c = Counter()
    for t, task, src in CT:
        for e in ex:
            if e.source == src and e.task == task and e.label:
                c[(t, e.split, e.form)] += 1
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "& \\multicolumn{2}{c}{train} & \\multicolumn{2}{c}{dev} & \\multicolumn{2}{c}{test} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Task & Ar. & Az. & Ar. & Az. & Ar. & Az. \\\\", "\\midrule"]
    for t in TASKS:
        L.append(TN[t] + " & " + " & ".join(str(c[(t, s, f)]) for s in ("train", "dev", "test") for f in ("arabic", "arabizi")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("data_stats", L)
    for t in TASKS:
        mac(f"nTest{t.replace('_', '')}Az", c[(t, 'test', 'arabizi')], "{}")
    # pair counts
    from formal_si.confirm import pair_sets
    from formal_si.geometry import tarc_pairs
    trdev = [e for e in ex if e.split != "test"]
    groups, _ = pair_sets(trdev)
    mac("nPairs", len(groups), "{}")
    tall = tarc_pairs(trdev, "train")
    mac("nTarcAvail", len(tall), "{:,}")


# ---------------- pair sources ----------------
def pair_table():
    L = ["\\begin{tabular}{llrr}", "\\toprule",
         "Source & Arabic side from & train$^{\\dagger}$ & dev \\\\", "\\midrule",
         f"human (S1) & NArabizi annotators & 0 & 0 \\\\",
         f"rule (S2) & 41 rewrite rules & {TQ['train_cross_fitted_cer_rule']:.2f} & {TQ['dev_cer_rule']:.2f} \\\\",
         f"dict.\\ (S3) & token dictionary & {TQ['train_cross_fitted_cer_dict']:.2f} & {TQ['dev_cer_dict_full']:.2f} \\\\",
         f"neural (S3c)$^{{\\ast}}$ & dict.\\ + char.\\ seq2seq & {NQ['train_cross_fitted_cer_neural_hybrid']:.2f} & {NQ['dev_cer_neural_hybrid_full']:.2f} \\\\",
         f"TArC (S1T)$^{{\\ast}}$ & TArC annotators & -- & -- \\\\",
         "\\bottomrule", "\\end{tabular}"]
    write("pair_sources", L)
    mac("cerRule", TQ["train_cross_fitted_cer_rule"]); mac("cerDict", TQ["train_cross_fitted_cer_dict"])
    mac("cerNeural", NQ["train_cross_fitted_cer_neural_hybrid"]); mac("cerNeuralDev", NQ["dev_cer_neural_hybrid_full"])
    mac("cerDictDev", TQ["dev_cer_dict_full"]); mac("cerRuleDev", TQ["dev_cer_rule"])


# ---------------- main results (test) ----------------
def main_results():
    S = ST["summary"]
    syst = [("S0_raw", "Raw"), ("S1_human", "Human"), ("S2_rule_pairs", "Rule p."), ("S3_dict_pairs", "Dict. p."),
            ("S4_derom_rule", "Rule in."), ("S4b_derom_dict", "Dict. in.")]
    L = ["\\begin{tabular}{ll r rrrr rr}", "\\toprule",
         " & & & \\multicolumn{4}{c}{remove $P_{16}$ (pairs from)} & \\multicolumn{2}{c}{transliterate input} \\\\",
         "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
         "Task & Encoder & In-scr. & none & human & rule & dict. & rule & dict. \\\\", "\\midrule"]
    for t in TASKS:
        for j, e in enumerate(ENCS):
            vals = [S[f"{t}|{e}|{s}"]["cross_script"] for s, _ in syst]
            best = max(vals)
            cv = [("\\textbf{%.1f}" % v if v == best else "%.1f" % v) for v in vals]
            L.append(f"{TN[t] if j == 0 else ''} & {EN[e]} & {S[f'{t}|{e}|S0_raw']['in_script']:.1f} & " + " & ".join(cv) + " \\\\")
        ch = [S[f"{t}|none|{s}"]["cross_script"] for s in ("S6_char_raw", "S5_char_rule", "S5b_char_dict")]
        L.append(f" & \\textit{{char n-gram}} & {S[f'{t}|none|S6_char_raw']['in_script']:.1f} & {ch[0]:.1f} & -- & -- & -- & {ch[1]:.1f} & {ch[2]:.1f} \\\\")
        L.append("\\midrule" if t != TASKS[-1] else "\\bottomrule")
    L.append("\\end{tabular}")
    write("test_main", L)
    # in-script cost / code-switch appendix table
    L = ["\\begin{tabular}{ll rrr rrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{in-script F1} & \\multicolumn{3}{c}{code-switched F1} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "Task & Encoder & none & human & dict. in. & none & human & dict. in. \\\\", "\\midrule"]
    costs = []
    for t in TASKS:
        for j, e in enumerate(ENCS):
            r = [S[f"{t}|{e}|{s}"] for s in ("S0_raw", "S1_human", "S4b_derom_dict")]
            costs.append(r[1]["in_script"] - r[0]["in_script"])
            cs = [f"{x['codeswitch']:.1f}" if "codeswitch" in x else "--" for x in r]
            L.append(f"{TN[t] if j == 0 else ''} & {EN[e]} & " + " & ".join(f"{x['in_script']:.1f}" for x in r) + " & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("test_inscript", L)
    gaps = [S[f"{t}|{e}|S0_raw"]["in_script"] - S[f"{t}|{e}|S0_raw"]["cross_script"] for t in TASKS for e in ENCS]
    mac("gapMin", min(gaps), "{:.1f}"); mac("gapMax", max(gaps), "{:.1f}"); mac("gapMean", np.mean(gaps), "{:.1f}")
    cs_d = [S[f"{t}|{e}|S1_human"]["codeswitch"] - S[f"{t}|{e}|S0_raw"]["codeswitch"] for t in ("narabizi_sent", "narabizi_topic") for e in ENCS]
    mac("csDropMax", -min(cs_d), "{:.1f}")
    for sy, nm in (("S1_human", "Human"), ("S4b_derom_dict", "DictIn"), ("S3_dict_pairs", "DictP"), ("S2_rule_pairs", "RuleP"), ("S4_derom_rule", "RuleIn")):
        mac(f"meanIn{nm}", np.mean([S[f"{t}|{e}|{sy}"]["in_script"] - S[f"{t}|{e}|S0_raw"]["in_script"] for t in TASKS for e in ENCS]))
        mac(f"meanCross{nm}", np.mean([S[f"{t}|{e}|{sy}"]["cross_script"] - S[f"{t}|{e}|S0_raw"]["cross_script"] for t in TASKS for e in ENCS]))
    mac("inCostMean", np.mean(costs)); mac("inCostMin", min(costs), "{:.1f}"); mac("inCostMax", max(costs), "{:.1f}")
    mac("charDictNar", S["narabizi_sent|none|S5b_char_dict"]["cross_script"], "{:.1f}")
    mac("charDictTsac", S["tsac_sent|none|S5b_char_dict"]["cross_script"], "{:.1f}")
    mac("charDictMyc", S["myc_sent|none|S5b_char_dict"]["cross_script"], "{:.1f}")
    best_enc = {t: max(S[f"{t}|{e}|{s}"]["cross_script"] for e in ENCS for s in ("S1_human", "S4b_derom_dict")) for t in TASKS}
    char_best = {t: max(S[f"{t}|none|{s}"]["cross_script"] for s in ("S5_char_rule", "S5b_char_dict", "S6_char_raw")) for t in TASKS}
    mac("nTasksCharWins", sum(char_best[t] > best_enc[t] for t in TASKS), "{}")


# ---------------- hypotheses: frozen + dependency-aware ----------------
HYP = [("H1 S1>S0", "H1 human vs.\\ none"), ("H2 S1>S2", "H2 human vs.\\ rule p."), ("H3 S1>S3", "H3 human vs.\\ dict.\\ p."),
       ("sec S1 vs S4", "human p.\\ vs.\\ rule input"), ("sec S1 vs S4b", "human p.\\ vs.\\ dict.\\ input")]


def hyp_tables():
    H = ST["hypotheses"]
    L = ["\\begin{tabular}{l r c c c c c}", "\\toprule",
         " & \\multicolumn{3}{c}{frozen analysis (12 cells)} & \\multicolumn{3}{c}{dependency-aware (joint bootstrap)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Comparison & $\\bar\\Delta$ & 95\\% CI & Holm & corpus-weighted & TSAC+MYC & tasks$^{+}$ \\\\", "\\midrule"]
    for k, n in HYP:
        v, d = H[k], DEP[k]
        cw, oo = d["corpus_weighted_mean"], d["out_of_narabizi_mean"]
        L.append(f"{n} & {sgn(v['mean_delta'])} & {ci(v['agg_lo'], v['agg_hi'])} & {int(v['n_cells_holm_sig'])}/12 & "
                 f"{sgn(cw['observed'])} {ci(cw['lo'], cw['hi'])} & {sgn(oo['observed'])} {ci(oo['lo'], oo['hi'])} & "
                 f"{d['n_tasks_holm_sig']}/4 \\\\")
        if k == "H3 S1>S3":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("test_hyp", L)
    for k, tag in (("H1 S1>S0", "One"), ("H2 S1>S2", "Two"), ("H3 S1>S3", "Three"), ("sec S1 vs S4", "SFour"), ("sec S1 vs S4b", "SFourb")):
        v, d = H[k], DEP[k]
        mac(f"h{tag}", v["mean_delta"]); mac(f"h{tag}lo", v["agg_lo"]); mac(f"h{tag}hi", v["agg_hi"])
        mac(f"h{tag}holm", int(v["n_cells_holm_sig"]), "{}")
        mac(f"h{tag}jlo", d["cell_mean_joint"]["lo"]); mac(f"h{tag}jhi", d["cell_mean_joint"]["hi"])
        for key, nm in (("corpus_weighted_mean", "cw"), ("out_of_narabizi_mean", "oo")):
            mac(f"h{tag}{nm}", d[key]["observed"]); mac(f"h{tag}{nm}lo", d[key]["lo"]); mac(f"h{tag}{nm}hi", d[key]["hi"])
        for c in ("narabizi", "tsac", "myc"):
            r = d["corpora"][c]
            mac(f"h{tag}{c}", r["observed"]); mac(f"h{tag}{c}lo", r["lo"]); mac(f"h{tag}{c}hi", r["hi"])
        mac(f"h{tag}tasksholm", d["n_tasks_holm_sig"], "{}")
        lo = min(r["lo"] for r in d["leave_one_encoder_out"].values())
        mac(f"h{tag}loeomin", lo)
        mac(f"h{tag}lotomin", min(r["observed"] for r in d["leave_one_task_out"].values()))
        mac(f"h{tag}lotominlo", min(r["lo"] for r in d["leave_one_task_out"].values()))
    # per-task table (appendix)
    L = ["\\begin{tabular}{l ccc}", "\\toprule", "Task (mean of 3 encoders) & H1 & H2 & H3 \\\\", "\\midrule"]
    for t in TASKS:
        cells = []
        for k in ("H1 S1>S0", "H2 S1>S2", "H3 S1>S3"):
            r = DEP[k]["tasks"][t]
            star = "$^{*}$" if r["p_holm_4tasks"] < 0.05 else ""
            cells.append(f"{sgn(r['observed'], 1)} {ci(r['lo'], r['hi'], 1)}{star}")
        L.append(f"{TN[t]} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule"]
    for e in ENCS:
        L.append(f"leave out {EN[e]} & " + " & ".join(
            f"{sgn(DEP[k]['leave_one_encoder_out'][e]['observed'], 1)} {ci(DEP[k]['leave_one_encoder_out'][e]['lo'], DEP[k]['leave_one_encoder_out'][e]['hi'], 1)}"
            for k in ("H1 S1>S0", "H2 S1>S2", "H3 S1>S3")) + " \\\\")
    for t in TASKS:
        L.append(f"leave out {TN[t]} & " + " & ".join(
            f"{sgn(DEP[k]['leave_one_task_out'][t]['observed'], 1)} {ci(DEP[k]['leave_one_task_out'][t]['lo'], DEP[k]['leave_one_task_out'][t]['hi'], 1)}"
            for k in ("H1 S1>S0", "H2 S1>S2", "H3 S1>S3")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("test_tasks", L)
    # per-cell (appendix)
    L = ["\\begin{tabular}{llccc}", "\\toprule", "Task & Encoder & H1 & H2 & H3 \\\\", "\\midrule"]
    for t in TASKS:
        for j, e in enumerate(ENCS):
            cs = []
            for k in ("H1 S1>S0", "H2 S1>S2", "H3 S1>S3"):
                v = H[k]["cells"][f"{t}|{e}"]
                cs.append(f"{sgn(v['delta'], 1)} {ci(v['lo'], v['hi'], 1)}{'$^{*}$' if v['p_holm'] < 0.05 else ''}")
            L.append(f"{TN[t] if j == 0 else ''} & {EN[e]} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("test_cells", L)
    # dev hypotheses
    Hd = SD["hypotheses"]
    L = ["\\begin{tabular}{lrc c}", "\\toprule", "Comparison (dev) & $\\bar\\Delta$ & 95\\% CI & Holm \\\\", "\\midrule"]
    for k, n in HYP:
        v = Hd[k]
        L.append(f"{n} & {sgn(v['mean_delta'])} & {ci(v['agg_lo'], v['agg_hi'])} & {int(v['n_cells_holm_sig'])}/12 \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("dev_hyp", L)
    mac("devHThree", Hd["H3 S1>S3"]["mean_delta"])
    S = ST["summary"]
    inn = [H["H1 S1>S0"]["cells"][f"{t}|{e}"]["delta"] for t in ("narabizi_sent", "narabizi_topic") for e in ENCS]
    out = [H["H1 S1>S0"]["cells"][f"{t}|{e}"]["delta"] for t in ("tsac_sent", "myc_sent") for e in ENCS]
    mac("hOneIn", np.mean(inn)); mac("hOneOut", np.mean(out))


# ---------------- learning curve ----------------
def curve_table():
    V = json.load(open(prefer_out("figures/fig_H_curve_values.json")))
    L = ["\\begin{tabular}{l rrrrr}", "\\toprule", "Pairs from & 50 & 100 & 250 & 500 & 998 \\\\", "\\midrule"]
    for s, n in (("human", "human"), ("dict", "dictionary"), ("rule", "rule")):
        c = V["curve"][s]
        L.append(n + " & " + " & ".join(f"{m:.1f} [{a:.1f}, {b:.1f}]" if i < 4 else f"{m:.1f}"
                                         for i, (m, a, b) in enumerate(zip(c["mean"], c["min_draw"], c["max_draw"]))) + " \\\\")
    L += ["\\midrule", f"no removal & \\multicolumn{{5}}{{c}}{{{V['no_removal_dev']:.1f}}} \\\\", "\\bottomrule", "\\end{tabular}"]
    write("curve", L)
    mac("curveHumanFifty", V["curve"]["human"]["mean"][0], "{:.1f}")
    mac("curveHumanFull", V["curve"]["human"]["mean"][-1], "{:.1f}")
    mac("curveRuleFull", V["curve"]["rule"]["mean"][-1], "{:.1f}")
    mac("curveDictFull", V["curve"]["dict"]["mean"][-1], "{:.1f}")
    mac("curveNone", V["no_removal_dev"], "{:.1f}")


# ---------------- geometry ----------------
def geometry_table():
    L = ["\\begin{tabular}{ll rrr rrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{spectrum of $S$} & \\multicolumn{2}{c}{held-out energy removed} & probe \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-8}",
         "Encoder & Pairs & top-16 & eff.\\ rank & PR & NArabizi dev & TArC dev & acc. \\\\", "\\midrule"]
    for e in ENCS:
        g = G[e]
        for j, s in enumerate(("human", "rule", "dict", "tarc_human")):
            sp = g["spectra"][s]
            h1 = g["heldout_energy_removed"]["narabizi_dev"][s]["energy_removed"]
            h2 = g["heldout_energy_removed"]["tarc_dev"][s]["energy_removed"]
            pr = g["script_probe"].get(s, None)
            nm = {"human": "human (NArabizi)", "rule": "rule", "dict": "dictionary", "tarc_human": "human (TArC)"}[s]
            L.append(f"{EN[e] if j == 0 else ''} & {nm} & {sp['top16_share']:.2f} & {sp['effective_rank']:.0f} & "
                     f"{sp['participation_ratio']:.0f} & {h1:.2f} & {h2:.2f} & {pr if pr is None else f'{pr:.2f}'} \\\\".replace("None", "--"))
        L.append(f" & none & & & & 0 & 0 & {g['script_probe']['none']:.2f} \\\\")
        L.append("\\midrule" if e != ENCS[-1] else "\\bottomrule")
    L.append("\\end{tabular}")
    write("geometry", L)
    L = ["\\begin{tabular}{l rrr rr}", "\\toprule",
         " & \\multicolumn{3}{c}{overlap $\\|U_a^\\top U_b\\|_F^2/k$} & \\multicolumn{2}{c}{stability (human)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
         "Encoder & hum.--dict. & hum.--rule & NArabizi--TArC & $n{=}50$ & split-half \\\\", "\\midrule"]
    for e in ENCS:
        g = G[e]
        L.append(f"{EN[e]} & {g['subspaces']['dict|human']['overlap']:.2f} & {g['subspaces']['human|rule']['overlap']:.2f} & "
                 f"{g['subspaces']['human|tarc_human']['overlap']:.2f} & {g['stability']['50']['mean']:.2f} & {g['stability']['split_half']:.2f} \\\\")
    L += ["\\midrule", "random $k$-dim.\\ subspace & \\multicolumn{3}{c}{" + f"{G['e5-small']['random_overlap_expected']:.3f} (mE5), {G['labse']['random_overlap_expected']:.3f} (others)" + "} & & \\\\",
          "\\bottomrule", "\\end{tabular}"]
    write("subspaces", L)
    ov = lambda k: [G[e]["subspaces"][k]["overlap"] for e in ENCS]
    mac("ovHDmin", min(ov("dict|human"))); mac("ovHDmax", max(ov("dict|human")))
    mac("ovHRmin", min(ov("human|rule"))); mac("ovHRmax", max(ov("human|rule")))
    mac("ovHTmin", min(ov("human|tarc_human"))); mac("ovHTmax", max(ov("human|tarc_human")))
    E16 = [G[e]["task_overlap"][t]["arabic"]["E_16_human"] for e in ENCS for t in TASKS]
    mac("EsixteenMin", min(E16)); mac("EsixteenMax", max(E16))
    mac("eighMax", max(max(G[e]["timing_eigh_s"].values()) for e in ENCS))
    mac("projMax", max(G[e]["timing_project_10k_s"] for e in ENCS))
    mac("probeNoneMin", min(G[e]["script_probe"]["none"] for e in ENCS))
    mac("probeHumanMin", min(G[e]["script_probe"]["human"] for e in ENCS))
    mac("probeHumanMax", max(G[e]["script_probe"]["human"] for e in ENCS))
    rng = lambda xs, f="{:.2f}": (f.format(min(xs)), f.format(max(xs)))
    allsrc = ("human", "rule", "dict", "tarc_human")
    a, b = rng([100 * G[e]["spectra"][s]["top16_share"] for e in ENCS for s in allsrc], "{:.0f}"); mac("topShareMin", a); mac("topShareMax", b)
    hd = lambda dev, src: [G[e]["heldout_energy_removed"][dev][src]["energy_removed"] for e in ENCS]
    for dev, dn in (("narabizi_dev", "Nar"), ("tarc_dev", "Tarc")):
        for src, sn in (("human", "Human"), ("tarc_human", "TarcPairs"), ("rule", "Rule"), ("dict", "Dict")):
            a, b = rng(hd(dev, src)); mac(f"held{dn}{sn}Min", a); mac(f"held{dn}{sn}Max", b)
    a, b = rng([G[e]["stability"]["50"]["mean"] for e in ENCS]); mac("stabFiftyMin", a); mac("stabFiftyMax", b)
    a, b = rng([G[e]["stability"]["split_half"] for e in ENCS]); mac("splitHalfMin", a); mac("splitHalfMax", b)
    for k, nm in (("dict|human", "HD"), ("human|rule", "HR"), ("human|tarc_human", "HT")):
        a, b = rng([sum(x < 45 for x in G[e]["subspaces"][k]["angles_deg"]) for e in ENCS], "{}"); mac(f"ang{nm}Min", a); mac(f"ang{nm}Max", b)
    dim = {"e5-small": 384, "labse": 768, "xlmr-base": 768}
    a, b = rng([G[e]["task_overlap"][t]["arabic"]["E_16_human"] / (16 / dim[e]) for e in ENCS for t in TASKS], "{:.0f}")
    mac("ratioMin", a); mac("ratioMax", b)
    J = json.load(open(prefer_out("figures/fig_J_values.json")))
    mac("rhoSel", J["E16_vs_selective"]["rho"]); mac("pSel", J["E16_vs_selective"]["p"])
    mac("rhoCost", J["E16_vs_inscript_cost"]["rho"]); mac("pCost", J["E16_vs_inscript_cost"]["p"])


# ---------------- selective objective (pilot, negative result) ----------------
def selective_table():
    PT = json.load(open(PKG / "reports/pilot/pilot_analysis.json"))["tests"]
    pm = {"narabizi_sent": "pilot_narabizi", "tsac_sent": "pilot_tsac", "myc_sent": "pilot_myc", "narabizi_topic": "pilot_narabizi_topic"}
    L = ["\\begin{tabular}{l ccc}", "\\toprule", "Task & mE5-small & LaBSE & XLM-R \\\\", "\\midrule"]
    ds = []
    for t in TASKS:
        cs = []
        for e in ENCS:
            v = PT[f"loco|{pm[t]}|{e}|selective (best lam0>0) vs invariance-only"]
            ds.append(v)
            cs.append(f"{sgn(v['delta'], 1)} {ci(v['lo'], v['hi'], 1)}")
        L.append(f"{TN[t]} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("selective", L)
    mac("selNS", sum(v["lo"] <= 0 <= v["hi"] for v in ds), "{}")
    mac("selWorse", sum(v["hi"] < 0 for v in ds), "{}")
    mac("selBetter", sum(v["lo"] > 0 for v in ds), "{}")


# ---------------- post-confirmatory v2 ----------------
def v2_tables(split):
    p = PKG / f"runs/confirm2/{split}/stats.json"
    if not p.exists():
        return
    R = json.load(open(p))
    C3, CL = R["contrasts"]["three_encoders"], R["contrasts"]["labse-st"]
    rows = [("E1 S1T>S0", "Tunisian (TArC) pairs vs.\\ none"), ("E1 S1T vs S1", "Tunisian vs.\\ Algerian human pairs"),
            ("E2 S1 vs S3c", "human vs.\\ neural-hybrid pairs"), ("E2 S3c vs S3", "neural-hybrid vs.\\ dict.\\ pairs"),
            ("E3 S4c vs S4b", "neural vs.\\ dict.\\ input"), ("E3 S1 vs S4c", "human pairs vs.\\ neural input")]
    L = ["\\begin{tabular}{l cccc c}", "\\toprule",
         "Contrast & NArabizi sent. & TSAC & MYC & NArabizi topic & 12-cell mean \\\\", "\\midrule"]
    for k, n in rows:
        r = C3[k]
        L.append(n + " & " + " & ".join(f"{sgn(r['tasks'][t]['observed'], 1)} {ci(r['tasks'][t]['lo'], r['tasks'][t]['hi'], 1)}" for t in TASKS)
                 + f" & {sgn(r['cell_mean_joint']['observed'], 1)} {ci(r['cell_mean_joint']['lo'], r['cell_mean_joint']['hi'], 1)} \\\\")
        tag = k.replace(" ", "").replace(">", "gt").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four")
        tag = "".join(ch for ch in tag if ch.isalpha())
        for t in TASKS:
            mac(f"v{split}{tag}{t.replace('_', '')}", r["tasks"][t]["observed"])
            mac(f"v{split}{tag}{t.replace('_', '')}lo", r["tasks"][t]["lo"]); mac(f"v{split}{tag}{t.replace('_', '')}hi", r["tasks"][t]["hi"])
        mac(f"v{split}{tag}mean", r["cell_mean_joint"]["observed"])
        mac(f"v{split}{tag}lo", r["cell_mean_joint"]["lo"]); mac(f"v{split}{tag}hi", r["cell_mean_joint"]["hi"])
    L += ["\\midrule", "\\multicolumn{6}{l}{\\textit{LaBSE with its sentence-transformers head (LaBSE-ST)}} \\\\"]
    for k, n in (("E4 H1 S1>S0", "H1 human vs.\\ none"), ("E4 H2 S1>S2", "H2 human vs.\\ rule p."), ("E4 H3 S1>S3", "H3 human vs.\\ dict.\\ p.")):
        r = CL[k]
        L.append(n + " & " + " & ".join(f"{sgn(r['tasks'][t]['observed'], 1)} {ci(r['tasks'][t]['lo'], r['tasks'][t]['hi'], 1)}" for t in TASKS)
                 + f" & {sgn(r['cell_mean_joint']['observed'], 1)} {ci(r['cell_mean_joint']['lo'], r['cell_mean_joint']['hi'], 1)} \\\\")
        tag = "LST" + k.split()[1]
        mac(f"v{split}{tag}mean".replace("1", "One").replace("2", "Two").replace("3", "Three"), r["cell_mean_joint"]["observed"])
        mac(f"v{split}{tag}lo".replace("1", "One").replace("2", "Two").replace("3", "Three"), r["cell_mean_joint"]["lo"])
        mac(f"v{split}{tag}hi".replace("1", "One").replace("2", "Two").replace("3", "Three"), r["cell_mean_joint"]["hi"])
    L += ["\\bottomrule", "\\end{tabular}"]
    write(f"v2_{split}", L)
    S = R["summary"]
    # LaBSE-ST vs LaBSE-CLS raw cross-script and in-script
    for t in TASKS:
        pass
    a = np.mean([S[f"{t}|labse-st|S0_raw"]["cross_script"] for t in TASKS])
    b = np.mean([S[f"{t}|labse|S0_raw"]["cross_script"] for t in TASKS])
    mac(f"v{split}LSTraw", a, "{:.1f}"); mac(f"v{split}LCLSraw", b, "{:.1f}")
    a = np.mean([S[f"{t}|labse-st|S0_raw"]["in_script"] for t in TASKS])
    b = np.mean([S[f"{t}|labse|S0_raw"]["in_script"] for t in TASKS])
    mac(f"v{split}LSTin", a, "{:.1f}"); mac(f"v{split}LCLSin", b, "{:.1f}")
    ch = {t: S[f"{t}|none|S5c_char_neural"]["cross_script"] for t in TASKS}
    for t in TASKS:
        mac(f"v{split}charNeural{t.replace('_', '')}", ch[t], "{:.1f}")


def compute_table():
    rows = []
    def meta(path):
        p = PKG / path
        return json.loads(p.read_text()) if p.exists() else None
    for enc in ENCS:
        m = meta(f"artifacts/encoding_meta/cache/{enc}/meta.json")
        rows.append((f"encode pilot strings, {EN[enc]}", m["n_texts"], m["seconds"], None))
    for root, lab in (("cache_confirm", "confirmatory"), ("cache_v2", "post-conf.")):
        for enc in ENCS:
            m = meta(f"artifacts/encoding_meta/{root}/{enc}/meta.json")
            if m:
                rows.append((f"encode {lab} strings, {EN[enc]}", m["n_texts"], m["seconds"], m.get("threads")))
    for path, lab in (("artifacts/encoding_meta/cache_labse_st/labse-st/meta.json", "encode all strings, LaBSE-ST"),
                      ("artifacts/encoding_meta/cache_v2/labse-st/meta.json", "encode post-conf.\\ strings, LaBSE-ST")):
        m = meta(path)
        if m:
            rows.append((lab, m["n_texts"], m["seconds"], m.get("threads")))
    rows.append(("train 3 neural transliterators", None, NQ["seconds_total"], NQ["threads"]))
    for path, lab in (("runs/confirm/dev/timing.json", "confirmatory evaluation, dev"), ("runs/confirm/test/timing.json", "confirmatory evaluation, test")):
        m = meta(path)
        rows.append((lab, None, m["seconds"], None))
    for split in ("dev", "test"):
        for p in sorted((PKG / f"runs/confirm2/{split}").glob("timing.*.json")) if (PKG / f"runs/confirm2/{split}").exists() else []:
            m = json.loads(p.read_text())
            rows.append((f"post-conf.\\ evaluation, {split} ({p.stem.split('.')[1]})", None, m["seconds"], None))
    L = ["\\begin{tabular}{lrrr}", "\\toprule", "Step & strings & wall s & threads \\\\", "\\midrule"]
    tot = 0.0
    for lab, n, sec, th in rows:
        tot += sec
        L.append(f"{lab} & {'' if n is None else f'{n:,}'} & {sec:,.0f} & {'n.r.' if th is None else th} \\\\")
    L += ["\\midrule", f"sum of listed steps & & {tot:,.0f} & \\\\", "\\bottomrule", "\\end{tabular}"]
    write("compute_rows", L)
    mac("computeListedHours", tot / 3600, "{:.1f}")


if HAVE_CORPUS:
    data_stats()
else:
    print("data/examples.jsonl not found: skipping data_stats (needs the rebuilt corpus)")
pair_table(); compute_table(); main_results(); hyp_tables(); curve_table(); geometry_table(); selective_table()
v2_tables("dev"); v2_tables("test")
(OUT / "numbers.tex").write_text("% generated by scripts/make_tables.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
print(len(M), "macros;", sorted(p.name for p in OUT.glob("*.tex")))
