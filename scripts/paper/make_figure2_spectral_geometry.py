"""Figure 2 (SCHEMATIC: constructed vectors in R^3, not measured embeddings).

One-dimensional analogue of P_k = I - U_k U_k^T with k = 1:
  (a) orthogonal decomposition x = P_1 x + u u^T x, with u the removed direction and span(u)^perp the retained plane;
  (b) two renderings f(a), f(b) of one sentence that differ mainly along u; after P_1 they are 0.10 apart
      instead of 1.05.

All geometry is computed in R^3 and drawn with one fixed orthographic camera, so every linear relation
(vector addition, projection, orthogonality) is preserved exactly in the drawing. Text is typeset by LaTeX with
the Libertine text and Libertinus math fonts of the acmart manuscript.

Run from FORMAL_TALLIP_V3/:  python scripts/make_figure2_spectral_geometry.py
Writes figures/figure2_spectral_geometry.pdf (vector) and, via pdftoppm, figures/figure2_spectral_geometry.png.
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "text.usetex": True, "font.family": "serif", "font.size": 9,
    "text.latex.preamble": r"\usepackage{libertine}\usepackage{libertinust1math}\usepackage{amsmath}",
})
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, ConnectionPatch
import numpy as np

from _paths import FIGURES
OUT = FIGURES
BLUE, BLUE_D, ORANGE, ORANGE_D = "#2a78d6", "#1a5aa6", "#eb6834", "#b2441a"
INK, GRAY, TEAL, TEAL_D = "#3d3c3a", "#6e6d69", "#1baf7a", "#0d6e4c"

# ------------------------------------------------------------------ geometry (exact, checked)
u = np.array([0.22, -0.14, 1.0]); u /= np.linalg.norm(u)          # removed direction (unit)
P = np.eye(3) - np.outer(u, u)                                      # P_1 = I - u u^T
assert np.allclose(P @ P, P) and np.allclose(P, P.T) and np.isclose(np.trace(P), 2.0)
w1 = np.cross(u, [0.0, 1.0, 0.0]); w1 /= np.linalg.norm(w1)        # orthonormal basis of the plane
w2 = np.cross(u, w1)
assert np.allclose([w1 @ u, w2 @ u, w1 @ w2], 0) and np.isclose(np.linalg.norm(w2), 1)

# screen-friendly orthonormal basis of the plane: h runs left-to-right on screen, g runs back-to-front
h = -(w1 + w2) / np.linalg.norm(w1 + w2); g = (w2 - w1) / np.linalg.norm(w2 - w1)
assert np.allclose([h @ u, g @ u, h @ g], 0) and np.isclose(h @ h, 1) and np.isclose(g @ g, 1)

# panel (a)
x = 1.15 * h + 0.30 * g + 1.0 * u
z = P @ x                                                           # retained part
r = np.outer(u, u) @ x                                              # removed part u u^T x
assert abs(z @ u) < 1e-12                                           # z lies in span(u)^perp
assert np.allclose(z + r, x)                                        # x = P_1 x + u u^T x
assert abs(z @ r) < 1e-12                                           # retained is orthogonal to removed
assert np.isclose(x @ x, z @ z + (u @ x) ** 2)                      # Pythagoras

# panel (b): Delta = alpha u + beta h with ||Delta|| = 1.05 and ||P_1 Delta|| = 0.10 exactly
beta = 0.10; alpha = np.sqrt(1.05 ** 2 - beta ** 2)
c = 0.30 * h - 0.30 * g                                                      # in-plane centre of the pair
fa = c + 0.55 * alpha * u + 0.5 * beta * h                          # Arabic script
fb = c - 0.45 * alpha * u - 0.5 * beta * h                          # Arabizi
pa, pb = P @ fa, P @ fb
D_before, D_after = np.linalg.norm(fa - fb), np.linalg.norm(pa - pb)
assert abs(pa @ u) < 1e-12 and abs(pb @ u) < 1e-12                  # projections on the plane
assert np.isclose(D_before, 1.05) and np.isclose(D_after, 0.10)
assert np.allclose(fa - pa, (u @ fa) * u) and np.allclose(fb - pb, (u @ fb) * u)   # guides parallel to u
print(f"checks passed: ||Delta||={D_before:.4f}, ||P1 Delta||={D_after:.4f}, z.u={z @ u:.1e}, z.r={z @ r:.1e}")

# ------------------------------------------------------------------ fixed orthographic camera
AZ, EL = np.deg2rad(-58), np.deg2rad(24)
R_ = np.array([-np.sin(AZ), np.cos(AZ), 0.0])                       # screen right
U_ = np.array([-np.sin(EL) * np.cos(AZ), -np.sin(EL) * np.sin(AZ), np.cos(EL)])   # screen up


def S(p):
    p = np.asarray(p, float)
    return np.array([p @ R_, p @ U_])


def depth_side(p):
    """+1 above the plane (u-side), -1 below."""
    return np.sign(np.asarray(p) @ u)


# 3D reference frame: an axis-aligned box [X0,X1] x [Y0,Y1] x [Z0,Z1] in the coordinates e1, e2, e3.
(X0, X1), (Y0, Y1), (Z0, Z1) = (-0.70, 1.50), (-1.45, 0.75), (-0.60, 1.30)
CORNERS = {(i, j, k): np.array([(X0, X1)[i], (Y0, Y1)[j], (Z0, Z1)[k]]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
VIEW = np.cross(R_, U_)                                             # unit vector towards the viewer
HIDDEN = min(CORNERS, key=lambda c: CORNERS[c] @ VIEW)              # corner farthest from the viewer
assert VIEW @ np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)]) > 0.999


def plane_z(x1, x2):
    """e3-coordinate of the retained plane u.p = 0 above the point (x1, x2)."""
    return -(u[0] * x1 + u[1] * x2) / u[2]


# the retained plane, clipped to the box (a parallelogram with vertical sides in e1,e2 coordinates)
plane3 = [np.array([a, b, plane_z(a, b)]) for a, b in [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]]
assert all(abs(p @ u) < 1e-12 and Z0 < p[2] < Z1 for p in plane3)
PLANE_FRONT = 0.5 * (plane3[0] + plane3[1])                         # midpoint of the front edge (e2 = Y0)
BOXC = "#c9c8c4"


def frame(ax):
    """Light wireframe box; edges at the hidden corner dashed; e1, e2, e3 drawn on three visible edges."""
    for c, pnt in CORNERS.items():
        for dim in range(3):
            if c[dim] == 1:
                continue
            if dim != 2 and c[2] == 1:                              # open-top box: no edges on the top face
                continue
            c2 = list(c); c2[dim] = 1; c2 = tuple(c2)
            hidden = HIDDEN in (c, c2)
            a, b = S(pnt), S(CORNERS[c2])
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BOXC, lw=0.6, ls=(0, (3, 2.5)) if hidden else "-", zorder=1)
    # coordinate axes on visible edges, arrowheads at the positive end, labels outside the box
    axes = [((0, 0, 0), (1, 0, 0), "e_1", np.array([0.02, -0.15])),
            ((1, 0, 0), (1, 1, 0), "e_2", np.array([0.14, -0.06])),
            ((0, 0, 0), (0, 0, 1), "e_3", np.array([-0.14, 0.0]))]
    for c0, c1, name, off in axes:
        a, b = S(CORNERS[c0]), S(CORNERS[c1])
        assert HIDDEN not in (c0, c1)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=GRAY, lw=0.9, shrinkA=0,
                                     shrinkB=0, zorder=1.5))
        ax.text(*(0.5 * (a + b) + off), f"${name}$", color=GRAY, fontsize=9, ha="center", va="center", zorder=9)


def base(ax, title):
    ax.set_aspect("equal"); ax.axis("off")
    frame(ax)
    ax.add_patch(Polygon([S(p) for p in plane3], closed=True, fc=TEAL, ec=TEAL, alpha=0.13, lw=0.9, zorder=2))
    ax.set_title(title, fontsize=9.5, pad=4)


def arrow(ax, p, q, color, lw=1.6, ls="-", z=5, head=9):
    if ls == "-":
        ax.add_patch(FancyArrowPatch(S(p), S(q), arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt"))
        return
    # dashed shaft with a solid arrowhead
    a, b = S(p), S(q); t = b - 0.11 * (b - a) / np.linalg.norm(b - a)
    ax.plot([a[0], t[0]], [a[1], t[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(t - 0.01 * (b - a) / np.linalg.norm(b - a), b, arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def seg(ax, p, q, color, lw=1.0, ls="-", z=4, alpha=1.0):
    a, b = S(p), S(q)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha, solid_capstyle="butt")


def label(ax, p, txt, offset, color, ha="left", va="center", fs=9.5, leader=False):
    """Text at screen point S(p)+offset (data units); optional thin leader line back to the point."""
    a = S(p); t = a + np.asarray(offset)
    ax.text(*t, txt, color=color, ha=ha, va=va, fontsize=fs, zorder=9)
    if leader:
        ax.annotate("", xy=a, xytext=t, arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=6, shrinkB=4),
                    zorder=8)


def triad(ax, origin2d, L=0.36):
    """Coordinate axes e1, e2, e3 drawn as a small separate triad (same camera)."""
    for v, name in zip(np.eye(3), ["e_1", "e_2", "e_3"]):
        d = S(v) * L
        ax.add_patch(FancyArrowPatch(origin2d, origin2d + d, arrowstyle="-|>", mutation_scale=6, color=GRAY,
                                     lw=0.8, shrinkA=0, shrinkB=0, zorder=3))
        n = d / np.linalg.norm(d)
        ax.text(*(origin2d + d + 0.11 * n), f"${name}$", color=GRAY, fontsize=8, ha="center", va="center")


_e = S(plane3[1]) - S(plane3[0])
EDGE_ANG = np.degrees(np.arctan2(_e[1], _e[0]))                     # labels run along the plane's front edge
PLANE_IN = 0.13 * np.array([-np.sin(np.radians(EDGE_ANG)), np.cos(np.radians(EDGE_ANG))])

fig, (A, B) = plt.subplots(1, 2, figsize=(6.5, 3.15), gridspec_kw=dict(wspace=0.08))
XL, YL = (-1.85, 2.0), (-1.50, 1.68)

# ------------------------------------------------------------------ panel (a)
base(A, r"(a) Orthogonal decomposition $x = P_1x + uu^{\top}x$")
O = np.zeros(3)
arrow(A, O, 1.2 * u, INK, lw=2.2, z=6)                                    # removed direction
label(A, 1.2 * u, r"removed direction $u$", (-0.05, 0.05), INK, ha="right", va="bottom")
arrow(A, O, z, BLUE, lw=1.6, ls=(0, (4, 2)), z=6)                         # retained part
arrow(A, O, x, BLUE_D, lw=1.9, z=7)                                       # original vector
seg(A, z, x, GRAY, lw=1.2, ls=(0, (1.2, 1.6)), z=5)                       # removed part, parallel to u
# right-angle marker at z between the removed part (u) and the retained vector (z)
q1, q2 = 0.17 * u, -0.17 * z / np.linalg.norm(z)
for p_, q_ in [(z + q1, z + q1 + q2), (z + q1 + q2, z + q2)]:
    seg(A, p_, q_, INK, lw=0.7, z=6)
A.scatter(*S(O), s=10, color=INK, zorder=8)
label(A, O, r"$0$", (-0.06, -0.05), INK, ha="right", va="top", fs=8.5)
label(A, x, r"$x$", (0.06, 0.06), BLUE_D, va="bottom", fs=10.5)
label(A, z, r"$z = P_1x$ (retained)", (0.02, -0.27), BLUE, ha="center", va="top")
mid = 0.5 * (z + x)
# removed-component label outside the box's right edge, joined to the segment by a thin leader
_m = S(mid); _t = np.array([S(CORNERS[(1, 1, 0)])[0] + 0.10, _m[1] + 0.02])
A.plot([_m[0] + 0.05, _t[0] - 0.03], [_m[1], _t[1]], color=GRAY, lw=0.5, zorder=4)
A.text(*(_t + np.array([0, 0.07])), r"$uu^{\top}x$", color=GRAY, ha="left", va="center", fontsize=9.5, zorder=9)
A.text(*(_t + np.array([0, -0.10])), r"(removed)", color=GRAY, ha="left", va="center", fontsize=8.5, zorder=9)
A.text(*(S(PLANE_FRONT) + PLANE_IN - 0.015 * _e / np.linalg.norm(_e)), r"retained plane $\operatorname{span}(u)^{\perp}=\operatorname{range}(P_1)$",
       color=TEAL_D, fontsize=7.2, ha="center", va="bottom", zorder=9, rotation=EDGE_ANG, rotation_mode="anchor")
A.set_xlim(*XL); A.set_ylim(*YL)

# ------------------------------------------------------------------ panel (b)
base(B, r"(b) Arabic-script and Arabizi pair, before and after $P_1$")
anchor = -1.20 * h + 0.40 * g                                            # where u is drawn in (b)
arrow(B, anchor, anchor + 0.85 * u, INK, lw=2.2, z=6)
label(B, anchor + 0.85 * u, r"$u$", (-0.07, 0.02), INK, ha="right", va="bottom", fs=10.5)
# projection guides (parallel to u): dotted, thin
seg(B, fa, pa, BLUE, lw=0.9, ls=(0, (1, 1.5)), z=4)
seg(B, fb, pb, ORANGE, lw=0.9, ls=(0, (1, 1.5)), z=1.5, alpha=0.9)        # below the plane: drawn under it
# original pair distance
B.scatter(*S(fa), s=46, color=BLUE, marker="o", edgecolor="white", lw=0.6, zorder=7)
B.scatter(*S(fb), s=52, color=ORANGE, marker="^", edgecolor="white", lw=0.6, zorder=7)
label(B, fa, r"Arabic script $f(a)$", (-0.02, 0.14), BLUE_D, ha="center", va="bottom")
label(B, fb, r"Arabizi $f(b)$", (0.0, -0.14), ORANGE_D, ha="center", va="top")
# brace-like distance marker to the left of the pair
off = np.array([-0.22, 0.0])
sa, sb = S(fa) + off, S(fb) + off
B.annotate("", xy=sa, xytext=sb, arrowprops=dict(arrowstyle="|-|, widthA=0.25, widthB=0.25", color=INK, lw=0.7, shrinkA=0, shrinkB=0), zorder=6)
B.text(*(0.5 * (sa + sb) + np.array([-0.13, 0.0])), r"$\|\Delta\| = %.2f$" % D_before, ha="right", va="center",
       fontsize=8.0, color=INK)
# projected points (open markers) on the plane
B.scatter(*S(pa), s=30, facecolor="white", edgecolor=BLUE, marker="o", lw=1.2, zorder=7)
B.scatter(*S(pb), s=34, facecolor="white", edgecolor=ORANGE, marker="^", lw=1.2, zorder=7)
# magnified inset of the projected pair
ctr = 0.5 * (S(pa) + S(pb)); mag = 5.0
ins_c = np.array([1.43, 0.50]); ins_r = 0.56
B.add_patch(Circle(ins_c, ins_r, fc="white", ec=GRAY, lw=0.7, zorder=8))
B.add_patch(Circle(ctr, 0.15, fc="none", ec=GRAY, lw=0.6, zorder=8))
v = ins_c - ctr; v /= np.linalg.norm(v)
B.plot(*zip(ctr + 0.15 * v, ins_c - ins_r * v), color=GRAY, lw=0.5, zorder=8)
ia, ib = ins_c + mag * (S(pa) - ctr), ins_c + mag * (S(pb) - ctr)
B.scatter(*ia, s=40, facecolor="white", edgecolor=BLUE, marker="o", lw=1.3, zorder=10)
B.scatter(*ib, s=46, facecolor="white", edgecolor=ORANGE, marker="^", lw=1.3, zorder=10)
B.plot(*zip(ia, ib), color=INK, lw=0.6, zorder=9)
# inset is an exact similarity of the screen positions: |ia - ib| = mag * |S(pa) - S(pb)|
assert np.isclose(np.linalg.norm(ia - ib), mag * np.linalg.norm(S(pa) - S(pb)))
assert np.allclose((ia - ib) / np.linalg.norm(ia - ib), (S(pa) - S(pb)) / np.linalg.norm(S(pa) - S(pb)))
B.text(*(ins_c + np.array([0, ins_r - 0.12])), r"zoom $\times %d$" % mag, ha="center", va="center", fontsize=7.3,
       color=GRAY, zorder=10)
B.text(*(ia + np.array([0, 0.13])), r"$P_1f(a)$", ha="center", va="bottom", fontsize=7.8, color=BLUE_D, zorder=10)
B.text(*(ib + np.array([0, 0.13])), r"$P_1f(b)$", ha="center", va="bottom", fontsize=7.8, color=ORANGE_D, zorder=10)
B.text(*(0.5 * (ia + ib) + np.array([0, -0.13])), r"$\|P_1\Delta\| = %.2f$" % D_after, ha="center", va="top",
       fontsize=7.8, color=INK, zorder=10)
B.text(*(S(PLANE_FRONT) + PLANE_IN + 0.06 * _e / np.linalg.norm(_e)), r"retained plane $\operatorname{span}(u)^{\perp}$", color=TEAL_D,
       fontsize=8.3, ha="center", va="bottom", zorder=9, rotation=EDGE_ANG, rotation_mode="anchor")
B.set_xlim(*XL); B.set_ylim(*YL)

# white halo behind every in-panel label so that frame lines and guides never run through text
import matplotlib.patheffects as pe
for ax_ in (A, B):
    for t_ in ax_.texts:
        t_.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

# ------------------------------------------------------------------ shared legend and schematic note
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BLUE_D, lw=1.8, label="original vector"),
     Line2D([], [], color=BLUE, lw=1.5, ls=(0, (4, 2)), label="retained (projected)"),
     Line2D([], [], color=INK, lw=2.0, label=r"removed direction $u$"),
     Line2D([], [], color=GRAY, lw=1.2, ls=(0, (1.2, 1.6)), label="removed component / projection guide"),
     Line2D([], [], ls="", marker="o", color=BLUE, label="Arabic script"),
     Line2D([], [], ls="", marker="^", color=ORANGE, label="Arabizi"),
     Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="after $P_1$ (open)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 0.045),
           handlelength=2.2, columnspacing=1.3)
fig.text(0.5, 0.005, r"\textit{Schematic: constructed vectors in $\mathbb{R}^3$, not measured embeddings}",
         ha="center", va="bottom", fontsize=8, color=GRAY)
fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.17)

OUT.mkdir(exist_ok=True)
pdf = OUT / "figure2_spectral_geometry.pdf"
fig.savefig(pdf)
subprocess.run(["pdftoppm", "-r", "400", "-png", "-singlefile", str(pdf), str(OUT / "figure2_spectral_geometry")],
               check=True)
print("wrote", pdf, "and figure2_spectral_geometry.png")
