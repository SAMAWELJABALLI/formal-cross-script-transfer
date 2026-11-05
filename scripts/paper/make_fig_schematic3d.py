"""Figure 2 (SCHEMATIC: constructed vectors, not data). One-dimensional analogue in R^3 of P_k = I - U_k U_k^T.
(a) decomposition x = P_1 x + u u^T x with u the removed direction and span(u)^perp the retained plane;
(b) two renderings of one sentence that differ mostly along u collapse onto nearby points of the plane.
Run: python scripts/make_fig_schematic3d.py  -> figures/fig_schematic3d.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})

u = np.array([0.18, -0.12, 0.976]); u /= np.linalg.norm(u)
P = np.eye(3) - np.outer(u, u)
assert np.allclose(P @ P, P) and np.allclose(P, P.T) and np.isclose(np.trace(P), 2)
e1 = np.cross(u, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)


def setup(ax, title):
    ax.set_proj_type("ortho"); ax.view_init(elev=16, azim=-62)
    s = 1.6
    corners = [c1 * s * e1 + c2 * s * e2 for c1, c2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    ax.add_collection3d(Poly3DCollection([corners], facecolor=AQUA, alpha=0.15, edgecolor=AQUA, lw=0.8))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor("#e3e3e3")
    ax.grid(False)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-0.6, 1.6); ax.set_box_aspect((1, 1, 0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("$e_1$", labelpad=-14); ax.set_ylabel("$e_2$", labelpad=-14); ax.set_zlabel("$e_3$", labelpad=-14)
    ax.set_title(title, fontsize=10, pad=-2)


def arrow(ax, p, q, color, lw=1.8, ls="-"):
    p, q = np.asarray(p, float), np.asarray(q, float)
    ax.plot(*zip(p, q), color=color, lw=lw, ls=ls)
    d = (q - p) / np.linalg.norm(q - p) * 0.13
    ax.quiver(*(q - d), *d, color=color, arrow_length_ratio=1.0, lw=lw)


fig = plt.figure(figsize=(7.6, 3.9))
# (a)
ax = fig.add_subplot(121, projection="3d")
setup(ax, "(a) $x = P_1x + uu^{\\top}x$")
O = np.zeros(3)
x = np.array([0.95, -0.55, 1.15]); z = P @ x
assert abs(z @ u) < 1e-12 and np.isclose(x @ x, z @ z + (u @ x) ** 2)
arrow(ax, O, 1.35 * u, INK, lw=2.0)
ax.text(*(1.42 * u + [-0.15, 0, 0.02]), "$u$", color=INK, fontsize=12)
arrow(ax, O, x, BLUE)
ax.text(*(x + [0.05, 0, 0.07]), "$x$", color=BLUE, fontsize=12)
arrow(ax, O, z, BLUE, lw=1.6, ls="--")
ax.text(*(z + [0.1, 0, -0.22]), "$z=P_1x$", color=BLUE, fontsize=11)
ax.plot(*zip(z, x), color=INK, lw=1.1, ls=(0, (3, 2)))
ax.text(*((z + x) / 2 + [0.1, 0, -0.05]), "$uu^{\\top}x$", color=INK, fontsize=11)
q1 = 0.13 * (x - z) / np.linalg.norm(x - z); q2 = -0.13 * z / np.linalg.norm(z)
ax.plot(*zip(z + q1, z + q1 + q2, z + q2), color=INK, lw=0.8)
fig.text(0.25, 0.1, "shaded plane $=\\mathrm{span}(u)^{\\perp}=\\mathrm{range}(P_1)$ (retained)", ha="center", color="#0d6e4c", fontsize=9)
# (b)
ax = fig.add_subplot(122, projection="3d")
setup(ax, "(b) a pair before and after $P_1$")
c0 = 0.75 * e1 + 0.35 * e2; a = c0 + 0.62 * u; b = c0 - 0.45 * u + np.array([0.08, -0.06, 0.0])
pa, pb = P @ a, P @ b
arrow(ax, O, 1.35 * u, INK, lw=2.0)
ax.text(*(1.42 * u + [-0.15, 0, 0.02]), "$u$", color=INK, fontsize=12)
ax.scatter(*a, s=42, color=BLUE, marker="o", depthshade=False)
ax.scatter(*b, s=50, color=ORANGE, marker="^", depthshade=False)
ax.plot(*zip(a, b), color=INK, lw=1.0)
ax.text(*(a + [0.08, 0, 0.05]), "$f(a)$ Arabic script", color=BLUE, fontsize=9.5)
ax.text(*(b + [0.1, 0, -0.08]), "$f(b)$ Arabizi", color="#b2441a", fontsize=9.5)
ax.scatter(*pa, s=40, facecolor="white", edgecolor=BLUE, marker="o", depthshade=False, lw=1.5)
ax.scatter(*pb, s=46, facecolor="white", edgecolor=ORANGE, marker="^", depthshade=False, lw=1.5)
ax.plot(*zip(a, pa), color=BLUE, lw=0.8, ls=":"); ax.plot(*zip(b, pb), color=ORANGE, lw=0.8, ls=":")
ax.text(*(pa + [0.12, 0.0, -0.05]), "$P_1f(a)\\approx P_1f(b)$", color=INK, fontsize=9.5)
fig.text(0.74, 0.1, f"$\\|\\Delta\\|={np.linalg.norm(a-b):.2f}$  $\\rightarrow$  $\\|P_1\\Delta\\|={np.linalg.norm(pa-pb):.2f}$",
         ha="center", fontsize=9.5, color=INK)
fig.text(0.5, 0.005, "Schematic: constructed vectors in $\\mathbb{R}^3$, not measured embeddings", ha="center", fontsize=8.5,
         color=INK, style="italic")
fig.subplots_adjust(0, 0.08, 1, 0.92, wspace=0.02)
from _paths import FIGURES
out = FIGURES
fig.savefig(out / "fig_schematic3d.pdf"); fig.savefig(out / "fig_schematic3d.png", dpi=150)
