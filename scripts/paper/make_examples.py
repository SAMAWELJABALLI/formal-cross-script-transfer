"""Figure B / Table: the same authentic NArabizi TRAIN sentences under each pair source.
Selection is mechanical (no cherry-picking): NArabizi train groups ordered by md5(group id); the first five
with 4-9 Arabizi tokens and no URL/username are taken. No glosses: the corpus layers available to us contain no
translation, and we do not provide unverified translations.
Level B only (needs the rebuilt corpus and cache_neural_translit/):  python scripts/paper/make_examples.py   (table compiled inside main.tex with LuaLaTeX)
"""
import hashlib
import json
from pathlib import Path

from formal_si.confirm import load
from formal_si.text.normalize import deromanize, normalize_arabic
from formal_si.translit import cer, cross_fitted, resources

from _paths import OUT
ex = load(include_test=False)
_, nar = resources(ex)
cf_dict = cross_fitted(ex)
cf_neur = json.loads(Path("cache_neural_translit/cross_fitted_pairs.json").read_text())
order = sorted(nar, key=lambda g: hashlib.md5(g.encode()).hexdigest())
pick = [g for g in order if 4 <= len(nar[g]["arabizi"].split()) <= 9 and "http" not in nar[g]["arabizi"]
        and "@" not in nar[g]["arabizi"]][:5]


def latin_runs(s):
    """Inside Arabic-script cells, typeset Latin runs (French words, apostrophes) with the Latin font."""
    import re
    return re.sub(r"([^\u0600-\u06FF\s]+(?:\s+[^\u0600-\u06FF\s]+)*)", lambda m: "\\foreignlanguage{english}{" + m.group(1) + "}", s)


def tex(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_").replace("$", "\\$")


rows, js = [], []
for i, g in enumerate(pick, 1):
    az, ar = nar[g]["arabizi"], nar[g]["arabic"]
    ref = normalize_arabic(ar)
    outs = {"rule": normalize_arabic(deromanize(az)), "dict": cf_dict[g], "neural": cf_neur[g]}
    js.append({"group": g, "arabizi": az, "human_arabic": ar, **outs, **{f"cer_{k}": cer(v, ref) for k, v in outs.items()}})
    rows.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{({i})}} \\texttt{{{tex(az)}}}}}\\\\")
    rows.append(f" & human & \\ar{{{latin_runs(tex(ar))}}} & -- \\\\")
    for k, lab in (("rule", "rule"), ("dict", "dictionary"), ("neural", "neural hybrid")):
        rows.append(f" & {lab} & \\ar{{{latin_runs(tex(outs[k]))}}} & {cer(outs[k], ref):.2f} \\\\")
    rows.append("\\addlinespace")
body = "\n".join(rows)
tab = "\\begin{tabular}{@{}lllr@{}}\n\\toprule\n & source & Arabic-script side & CER \\\\\n\\midrule\n" + body + \
    "\n\\bottomrule\n\\end{tabular}\n"
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "tables/examples.tex").write_text(tab)
(OUT / "tables/examples.json").write_text(json.dumps(js, ensure_ascii=False, indent=1))
print("examples:", [p["group"] for p in js])
