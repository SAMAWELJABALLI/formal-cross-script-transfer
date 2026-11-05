"""Level A: regenerate the paper's figures from archived results and compare with figures/.

Runs the figure scripts in scripts/paper/ (outputs under $FORMAL_OUT/figures), then:
  * compares the JSON values written alongside the figures (fig_H_curve_values.json, fig_J_values.json,
    PROVENANCE.json, reports/v3/axes_stats.json) with the archived copies, exactly;
  * compares every archived PNG with its regenerated counterpart pixel by pixel;
  * checks that every archived PDF was regenerated. PDF bytes are not compared: they embed a creation
    date and depend on the matplotlib / font versions.

figure2_spectral_geometry needs a LaTeX installation (text.usetex with libertine / libertinust1math) and
pdftoppm; it is skipped, and reported as such, when either is missing.

Usage:  python scripts/reproduce_figures.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path(os.environ.get("FORMAL_OUT", "out"))
SCRIPTS = ["make_figures.py", "make_figures_v3.py", "make_tables_v3.py", "make_fig_schematic3d.py"]
VALUES = {"figures/fig_H_curve_values.json": "figures/fig_H_curve_values.json",
          "figures/fig_J_values.json": "figures/fig_J_values.json",
          "figures/PROVENANCE.json": "figures/PROVENANCE.json",
          "reports/v3/axes_stats.json": "reports/v3/axes_stats.json"}


def png_diff(a: Path, b: Path) -> str:
    import matplotlib.image as mpimg
    x, y = mpimg.imread(a), mpimg.imread(b)
    if x.shape != y.shape:
        return f"size {x.shape[1]}x{x.shape[0]} vs {y.shape[1]}x{y.shape[0]}"
    d = np.abs(x.astype(float) - y.astype(float))
    return "identical pixels" if d.max() == 0 else f"max |diff| {d.max():.3f}, {np.mean(d.max(-1) > 0.02) * 100:.2f}% pixels differ > 0.02"


def main():
    env = {**os.environ, "FORMAL_OUT": str(OUT), "MPLBACKEND": "Agg"}
    latex = shutil.which("latex") and shutil.which("pdftoppm")
    scripts = SCRIPTS + (["make_figure2_spectral_geometry.py"] if latex else [])
    for s in scripts:
        print(f"$ python scripts/paper/{s}", flush=True)
        subprocess.run([sys.executable, f"scripts/paper/{s}"], check=True, env=env, stdout=subprocess.DEVNULL)
    ok = True
    print()
    for regen_rel, arch_rel in VALUES.items():
        same = (OUT / regen_rel).read_bytes() == Path(arch_rel).read_bytes()
        ok &= same
        print(f"[{'OK' if same else 'MISMATCH'}]  {arch_rel}  ({'identical bytes' if same else 'differs'})")
    for arch in sorted(Path("figures").glob("*/*")):
        if arch.suffix not in (".pdf", ".png"):
            continue
        regen = OUT / "figures" / arch.name
        if not regen.exists():
            skipped = arch.stem.startswith("figure2_spectral_geometry") and not latex
            ok &= skipped
            print(f"[{'SKIPPED' if skipped else 'MISSING'}]  {arch}" + ("  (needs LaTeX and pdftoppm)" if skipped else ""))
        elif arch.suffix == ".png":
            msg = png_diff(arch, regen)
            print(f"[{'OK' if msg == 'identical pixels' else 'CHECK'}]  {arch}  ({msg})")
        else:
            print(f"[OK]  {arch}  (regenerated; PDF bytes not compared)")
    print("\nFIGURES REGENERATED" + ("" if ok else " WITH PROBLEMS (see above)") +
          f"; outputs in {OUT / 'figures'}. 'CHECK' lines are pixel differences to inspect visually.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
