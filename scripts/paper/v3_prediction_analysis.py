"""V3 analysis-only re-reading of the SAVED test predictions (no refitting, no new test runs).

Outputs reports/v3/predictions_analysis.json with
  - per-class precision / recall / F1 (pooled over the 5 resamples) per task x encoder x system x eval set
  - resample-pooled confusion matrices
  - item-level error categories (majority vote over resamples): projection corrects / degrades,
    input transliteration wins, both fail
  - example items for the error table (NArabizi sentiment, training-licence caveat applies)
Usage:  python scripts/paper/v3_prediction_analysis.py   (writes $FORMAL_OUT/reports/v3/predictions_analysis.json)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _paths import REPORTS_V3, confirm2_prediction_lines

OUT = REPORTS_V3
TASKS = ["narabizi_sent", "tsac_sent", "myc_sent", "narabizi_topic"]
ENCS = ["e5-small", "labse", "xlmr-base"]


def load_preds():
    P = defaultdict(list)  # (task, enc, system) -> list of preds dicts per resample
    for f in ("runs/confirm/test/predictions.jsonl", "runs/confirm2/test/predictions.jsonl"):
        for line in (open(f) if f.startswith("runs/confirm/") else confirm2_prediction_lines("test")):
            r = json.loads(line)
            key = (r["task"], r["encoder"], r["system"])
            if f.endswith("confirm2/test/predictions.jsonl") and key in P and len(P[key]) >= 5 \
                    and r["system"] in ("S0_raw", "S1_human", "S3_dict_pairs", "S4b_derom_dict"):
                # confirm2 re-ran S0/S1/S3/S4b deterministically; keep the confirmatory copy, but check equality
                continue
            P[key].append(r["preds"])
    return P


def prf(gold, pred, labels):
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[c] = {"precision": 100 * pr, "recall": 100 * rc, "f1": 100 * f1, "support": sum(g == c for g in gold)}
    return out


def main():
    gold1 = json.load(open("runs/confirm/test/gold.json"))
    gold2 = json.load(open("runs/confirm2/test/gold.json"))
    assert all(gold1[t][n]["labels"] == gold2[t][n]["labels"] for t in TASKS for n in gold1[t]), "gold mismatch"
    P = load_preds()
    # determinism check of re-run systems
    check = {}
    for line in confirm2_prediction_lines("test"):
        r = json.loads(line)
        if r["encoder"] in ENCS and r["system"] in ("S0_raw", "S1_human"):
            a = P[(r["task"], r["encoder"], r["system"])][r["resample"]]
            check[str((r["task"], r["encoder"], r["system"], r["resample"]))] = a == r["preds"]
    res = {"rerun_identical_fraction": float(np.mean(list(check.values()))) if check else None,
           "per_class": {}, "confusion": {}, "error_categories": {}, "examples": {}}
    systems = ["S0_raw", "S1_human", "S2_rule_pairs", "S3_dict_pairs", "S3c_neural_pairs", "S1T_tarc_pairs",
               "S4b_derom_dict", "S4c_derom_neural"]
    for t in TASKS:
        for ev, g in gold1[t].items():
            labels = sorted(set(g["labels"]))
            for (tt, enc, sysn), runs in P.items():
                if tt != t or ev not in runs[0]:
                    continue
                gold_pool = g["labels"] * len(runs)
                pred_pool = [p for r in runs for p in r[ev]]
                key = f"{t}|{enc}|{sysn}|{ev}"
                res["per_class"][key] = prf(gold_pool, pred_pool, labels)
                cm = np.zeros((len(labels), len(labels)))
                for a, b in zip(gold_pool, pred_pool):
                    cm[labels.index(a), labels.index(b)] += 1
                res["confusion"][key] = {"labels": labels, "matrix": (cm / len(runs)).round(2).tolist()}
    # item-level categories on cross-script rows: majority vote over resamples
    def vote(runs, ev):
        n = len(runs[0][ev])
        return [Counter(r[ev][i] for r in runs).most_common(1)[0][0] for i in range(n)]

    for t in TASKS:
        g = gold1[t]["cross_script"]["labels"]
        for enc in ENCS:
            s0, s1 = vote(P[(t, enc, "S0_raw")], "cross_script"), vote(P[(t, enc, "S1_human")], "cross_script")
            s4c = vote(P[(t, enc, "S4c_derom_neural")], "cross_script")
            cat = Counter()
            items = defaultdict(list)
            for i, y in enumerate(g):
                a, b, c = s0[i] == y, s1[i] == y, s4c[i] == y
                cat["n"] += 1
                if not a and b:
                    cat["projection_corrects"] += 1; items["projection_corrects"].append(i)
                if a and not b:
                    cat["projection_degrades"] += 1; items["projection_degrades"].append(i)
                if c and not b:
                    cat["input_translit_wins"] += 1; items["input_translit_wins"].append(i)
                if b and not c:
                    cat["projection_wins_over_input"] += 1; items["projection_wins_over_input"].append(i)
                if not a and not b and not c:
                    cat["all_fail"] += 1; items["all_fail"].append(i)
            res["error_categories"][f"{t}|{enc}"] = dict(cat)
            res["examples"][f"{t}|{enc}"] = {k: v for k, v in items.items()}
            res["examples"][f"{t}|{enc}"]["votes"] = {"S0": s0, "S1": s1, "S4c": s4c}
    (OUT / "predictions_analysis.json").write_text(json.dumps(res))
    print("rerun identical:", res["rerun_identical_fraction"])
    for k, v in res["error_categories"].items():
        print(k, v)


if __name__ == "__main__":
    main()
