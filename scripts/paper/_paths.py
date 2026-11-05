"""Path conventions shared by the paper scripts.

The scripts run from the repository root (they chdir there on import), read archived inputs from
runs/, reports/, artifacts/ and (for corpus-dependent parts only) data/, and write every output under
$FORMAL_OUT (default: out/) so that archived files are never overwritten:

    $FORMAL_OUT/tables/*.tex    $FORMAL_OUT/figures/*    $FORMAL_OUT/reports/v3/*.json
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path(os.environ.get("FORMAL_OUT", "out"))
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS_V3 = OUT / "reports" / "v3"
for _d in (TABLES, FIGURES, REPORTS_V3):
    _d.mkdir(parents=True, exist_ok=True)

# Corpus-dependent steps (data statistics, example sentences) need the locally rebuilt corpus,
# which is not distributed (see docs/DATASETS.md).
HAVE_CORPUS = Path("data/examples.jsonl").exists()
# The neural-transliteration text maps are derived corpus text and are not distributed either.
HAVE_NEURAL_MAPS = Path("cache_neural_translit/full_input_map.json").exists()


def prefer_out(rel: str) -> Path:
    """An input produced by an earlier paper script: the regenerated copy if present, else the archived one."""
    regenerated = OUT / rel
    if regenerated.exists():
        return regenerated
    return Path(rel)


def confirm2_prediction_lines(split: str):
    """Lines of runs/confirm2/<split>/predictions.jsonl.

    The original run directory also held predictions.jsonl, the concatenation of the per-part files in
    sorted order (written by `formal_si.confirm2 stats`). It is a byte-for-byte duplicate and is not
    archived; this function yields the same lines in the same order from the parts.
    """
    for part in sorted(Path(f"runs/confirm2/{split}").glob("predictions.*.jsonl")):
        with open(part) as fh:
            yield from fh
