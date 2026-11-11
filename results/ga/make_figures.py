"""Geographical Analysis paper: generation of Figs. 1–7 (reads the experiment output,
writes vector PDF and 300dpi PNG).

    PYTHONPATH=src python results/ga/make_figures.py

  Fig.1 study area and co-location structure (map, reads `data/` and `realdata/` directly)
  Fig.2 data flow         Fig.3 permutation validity
  Fig.4 permutation time  Fig.5 speed-up
  Fig.6 memory and the feasible-scale criterion    Fig.7 parallel scaling

Figure numbers were reassigned on 2026-10-14: the new map takes Fig.1 and the former
Fig.1–6 shift to Fig.2–7.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 120, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
})

C_SPARSE = "#c0392b"      # red: sparse path
C_REF = "#1f4e79"         # dark blue: reference O(N^2)
C_ACC = "#e08214"         # orange: accent
C_GREY = "#7f8c8d"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  -> {FIG / name}.pdf / .png")


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fig.2 data flow (algorithmic view)
# ---------------------------------------------------------------------------
def fig2_dataflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#eef3f9", ec="#1f4e79", ts=8.4, bs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color="#123")
        ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top",
                fontsize=bs, color="#334", linespacing=1.5)

    def arrow(p, q, color="#333", style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                     lw=1.15, color=color))

    ytop, htop = 3.10, 1.62
    ybot, hbot = 0.35, 1.75
    box(0.10, ytop, 2.6, htop, "Inputs  (once)",
        "coords $x \\in \\mathbf{R}^{N\\times 2}$\n"
        "types $t \\in \\{1..K\\}$\nperiods $\\pi \\in \\{1..P\\}$", fc="#f7f9fb")
    box(3.00, ytop, 2.8, htop, "Neighborhood → CSR",
        "KD-Tree radius query\n$r = c\\,b$   ($c = 3$)\n"
        "nnz $= \\rho N^{2}$ edges")
    box(6.10, ytop, 3.80, htop, "Invariant weights  (cached)",
        "$\\mathrm{data}_{ij} = w_s(i,j)\\,w_t(i,j)$\n"
        "independent of type labels\n⇒ built once, reused",
        fc="#fdf3e7", ec="#b9770e")

    box(0.10, ybot, 2.6, hbot, "Permutation  $\\sigma$",
        "row-permute design\nmatrix $Y$\nlast column $\\equiv 1$",
        fc="#eef7ee", ec="#1e8449")
    box(3.00, ybot, 2.8, hbot, "One SpMM",
        "$Z = W Y_\\sigma$\n$O(\\mathrm{nnz})$ per perm.\nCSR rows private")
    box(6.10, ybot, 3.80, hbot, "Reduce  $O(NK + K^{2})$",
        "$\\hat{C}[a,b] = Z_{ab} / Z_{aK} / p_b$\n"
        "numerator and row sums\nfrom the same SpMM",
        fc="#fdf3e7", ec="#b9770e")

    arrow((2.70, ytop + htop / 2), (3.00, ytop + htop / 2))
    arrow((5.80, ytop + htop / 2), (6.10, ytop + htop / 2))
    arrow((2.70, ybot + hbot / 2), (3.00, ybot + hbot / 2))
    arrow((5.80, ybot + hbot / 2), (6.10, ybot + hbot / 2))

    # vertical arrows: cache reuse is the core idea
    arrow((1.40, ytop), (1.40, ybot + hbot), color="#1e8449")
    ax.text(1.52, 2.62, "$\\times\\,n_{\\mathrm{perm}}$ iterations",
            ha="left", va="center", fontsize=6.9, color="#1e8449")
    arrow((4.40, ytop), (4.40, ybot + hbot), color=C_SPARSE)
    ax.text(4.52, 2.62, "reuse cached CSR\n(no rebuild, no copy)",
            ha="left", va="center", fontsize=6.9, color=C_SPARSE)
    arrow((8.00, ytop), (8.00, ybot + hbot), color="#b9770e")
    ax.text(8.12, 2.62, "$w_s\\!\\cdot\\!w_t$ fixed",
            ha="left", va="center", fontsize=6.9, color="#b9770e")

    ax.text(5.0, 4.95, "Structure-invariant sparse permutation inference: data flow",
            ha="center", fontsize=9.2, fontweight="bold")
    save(fig, "fig2_dataflow")


# ---------------------------------------------------------------------------
# Fig.4 / Fig.5 / Fig.6
# ---------------------------------------------------------------------------
def fig4_time(bench):
    """Fig.4: time per permutation against N (sparse O(nnz) vs reference O(N²))."""
    rows = bench["rows"]
    fits = bench["fits"]
    ns = np.array([r["n"] for r in rows], float)
    perm = np.array([r["perm_s"] for r in rows], float) * 1e3
    ref_n = np.array([r["n"] for r in rows if np.isfinite(r["ref_perm_s"])], float)
    ref_t = np.array([r["ref_perm_s"] for r in rows
                      if np.isfinite(r["ref_perm_s"])], float) * 1e3

    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.loglog(ns, perm, "o-", color=C_SPARSE, ms=4,
              label=f"sparse engine  (slope {fits['perm_slope']:.2f})")
    ax.loglog(ref_n, ref_t, "s--", color=C_REF, ms=4,
              label=f"reference $O(N^2)$  (slope {fits['ref_slope']:.2f})")
    x = np.array([ns.min(), ns.max()])
    ax.loglog(x, perm[0] / ns[0] * x, ":", color=C_SPARSE, lw=0.9, alpha=0.6,
              label="$O(\\mathrm{nnz})$ guide")
    ax.loglog(x, ref_t[0] / ref_n[0] ** 2 * x ** 2, ":", color=C_REF, lw=0.9,
              alpha=0.6, label="$O(N^2)$ guide")
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel("time per permutation (ms)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig4_permutation_time")


def fig5_speedup(bench):
    """Fig.5: speed-up against N (measured points plus power-law extrapolation)."""
    rows, fits = bench["rows"], bench["fits"]
    sp_n = np.array([r["n"] for r in rows if np.isfinite(r["speedup_vs_ref"])], float)
    sp = np.array([r["speedup_vs_ref"] for r in rows
                   if np.isfinite(r["speedup_vs_ref"])], float)

    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.loglog(sp_n, sp, "o-", color=C_ACC, ms=4.5, label="measured")
    for xi, yi in zip(sp_n, sp):
        ax.annotate(f"{yi:,.0f}×", (xi, yi), textcoords="offset points",
                    xytext=(4, 5), fontsize=6.8, color=C_ACC)

    # power-law extrapolation: the ratio of the two fitted lines gives
    # slope = ref_slope - perm_slope
    if np.isfinite(fits["ref_slope"]) and np.isfinite(fits["perm_slope"]):
        s = fits["ref_slope"] - fits["perm_slope"]
        x_all = np.array([sp_n.min(), 1e6])
        # anchor on the **measured speed-up at the largest N** (the last point where the
        # reference implementation can still be timed directly), not on the smallest N:
        # an extrapolation should start from the measured value nearest the target size.
        # the same token in render_manuscript.py uses this anchor; keep them in step.
        c = sp[-1] / sp_n[-1] ** s
        ax.loglog(x_all, c * x_all ** s, "--", color=C_GREY, lw=1.2,
                  label=f"power-law extrapolation (slope {s:.2f})")
        pred = c * 1e6 ** s
        # it ends top right under the legend, so put the label bottom right
        ax.annotate(f"$N=10^6$: {pred:,.0f}× (extrapolated)", (0.98, 0.04),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=6.8, color=C_GREY)
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel("speed-up vs. reference $O(N^2)$")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig5_speedup")


def fig6_memory(bench):
    rows = bench["rows"]
    ns = np.array([r["n"] for r in rows], float)
    rss = np.array([r["rss_mb"] for r in rows], float)
    dense = np.array([r["dense_mb_est"] for r in rows], float)
    csr = np.array([r["csr_mb_est"] for r in rows], float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.loglog(ns, rss, "o-", color=C_SPARSE, ms=4, label="measured peak RSS")
    ax.loglog(ns, csr, "^--", color="#1e8449", ms=4, label="CSR storage $16\\,\\mathrm{nnz}$")
    ax.loglog(ns, dense, "s:", color=C_REF, ms=4, label="dense $8N^2$ (hypothetical)")
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel("memory (MB)")
    ax.set_title("(a) Peak memory")
    ax.legend(loc="upper left", frameon=False)

    # (b) rho criterion: nnz = rho N^2
    from stclq.complexity import bandwidth_for_rho, feasible_scale_table
    ax = axes[1]
    side = 20000.0
    nn = np.logspace(3, 6, 40)
    for rho_max, style in ((0.01, "-"), (0.001, "--"), (0.0001, ":")):
        ax.loglog(nn, rho_max * nn ** 2, style, color=C_ACC if rho_max == 0.01 else C_GREY,
                  lw=1.4, label=f"$\\rho$ = {rho_max:g}")
    real = [r for r in rows]
    ax.loglog([r["n"] for r in real], [r["nnz"] for r in real], "o",
              color=C_SPARSE, ms=4.5, label="this study ($\\bar k \\approx 150$)")
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel("non-zeros  $\\mathrm{nnz} = \\rho N^2$")
    ax.set_title("(b) Feasible-scale criterion")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.06,
            f"$\\rho \\leq 0.01$ requires $b \\leq$ {bandwidth_for_rho(0.01, side**2):.0f} m\n"
            f"for a {side/1000:.0f} km $\\times$ {side/1000:.0f} km domain",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444")
    fig.tight_layout()
    save(fig, "fig6_memory_rho")
    return feasible_scale_table([10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7],
                                side=20000.0, rho_max=0.01, target_knn=150.0)


def fig7_parallel(par):
    strong = par["strong"]
    weak = par["weak"]
    platform = par.get("platform") or {}
    ps = np.array([s["p"] for s in strong], float)
    sp = np.array([s["speedup"] for s in strong], float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(ps, ps, ":", color=C_GREY, lw=1.2, label="ideal linear $S=p$")
    if platform.get("cpu"):
        cp = np.array([c["p"] for c in platform["cpu"]], float)
        cs = np.array([c["speedup"] for c in platform["cpu"]], float)
        ax.plot(cp, cs, "^--", color="#7d3c98", ms=3.6, lw=1.2,
                label="machine CPU-bound ceiling")
    ax.plot(ps, sp, "o-", color=C_SPARSE, ms=4, lw=1.4,
            label="permutation engine (measured)")
    if platform.get("max_bw_relative"):
        ax.axhline(platform["max_bw_relative"], color="#1e8449", ls="-.", lw=1.1,
                   label=(f"machine memory-bandwidth ceiling "
                          f"({platform['max_bw_relative']:.1f}×)"))
    ax.set_xlabel("processes $p$")
    ax.set_ylabel("speed-up $S(p)$")
    ax.set_title("(a) Strong scaling vs. platform ceilings")
    ax.set_xticks(ps)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    # keep a band at the bottom (no curve lies between 0.6 and 1.0) for the engine's
    # labels: this clears both the x ticks and the ideal line, which passes y=1 at p=1.
    ax.set_ylim(0.6, max(21.5, sp.max() * 1.15))
    for xi, yi in zip(ps, sp):
        ax.annotate(f"{yi:.1f}×", (xi, yi), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=6.8, color=C_SPARSE)

    ax = axes[1]
    wp = np.array([w["p"] for w in weak], float)
    weff = np.array([w["efficiency"] for w in weak], float)
    wn = np.array([w.get("n_perm", 0) for w in weak], float)
    ctrl = (platform.get("weak_control") or [])
    if not ctrl and platform.get("bandwidth"):
        # derive the platform's own memory-bound weak-scaling curve from the measured
        # per-process bandwidth; with constant work per process, efficiency = bw_p / bw_1
        bw1 = platform["bandwidth"][0]["per_process_gbs"]
        ctrl = [{"p": b["p"], "efficiency": b["per_process_gbs"] / bw1}
                for b in platform["bandwidth"] if b["per_process_gbs"] > 0]
    if ctrl:
        ax.plot([c["p"] for c in ctrl], [c["efficiency"] for c in ctrl], "^--",
                color="#7d3c98", ms=3.6, lw=1.2,
                label="platform control (memory-bound, same pattern)")
    ax.plot(wp, weff, "s-", color=C_REF, ms=4, label="permutation engine")
    ax.axhline(1.0, color=C_GREY, ls=":", lw=1)
    ax.set_xlabel("processes $p$  (with $n_{\\mathrm{perm}} \\propto p$)")
    ax.set_ylabel("efficiency  $T_1/T_p$")
    ax.set_ylim(0, 1.35)
    ax.set_xticks(wp)
    ax.set_title("(b) Weak scaling")
    if wn.max() > 0:
        for xi, ni in zip(wp, wn):
            ax.annotate(f"{int(ni)}", (xi, 0.035), ha="center",
                        fontsize=6.2, color="#666")
        # put the header at the top-left height so it does not share a row with a value
        ax.annotate("$n_{\\mathrm{perm}}$", (0.02, 0.10), xycoords="axes fraction",
                    ha="left", va="center", fontsize=6.2, color="#666")
    ax.legend(frameon=False, loc="upper right", fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig7_parallel_scaling")


def load_calibration():
    """Load the main calibration run (type-I error rate) and the optional power run.

    The main run ``calibration.json`` gives the type-I error rate of the H0 arm and the
    rejection rate of the H1 arm; ``calibration_power.json`` (produced by
    ``exp_power.py``) scans the clustering scale ``sep`` under a **moderate** alternative
    and yields a power curve with enough resolution to read.
    """
    cal = load_json(HERE / "calibration" / "calibration.json")
    pp = HERE / "calibration" / "calibration_power.json"
    cal["_power"] = load_json(pp) if pp.exists() else None
    return cal


def fig3_calibration(cal):
    rows = cal["rows"]
    alpha = cal["alpha"]
    power = cal.get("_power")
    deltas = sorted({r["delta"] for r in rows})
    present = {r["temporal"] for r in rows}
    temporals = [t for t in ("exponential", "inverse_dt", "none") if t in present]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    styles = {"perm:type": ("o", C_SPARSE, "global label permutation"),
              "perm:type_period": ("s", "#1e8449",
                                   "block (within-period) permutation"),
              "norm:type": ("^", C_ACC, "normal approx. (global)")}

    # ---- (a) H0: type-I error rate against composition imbalance delta ----
    ax = axes[0]
    for temporal, ls in zip(temporals, ("-", "--")):
        for key, (marker, color, label) in styles.items():
            xs, ys, es = [], [], []
            for d in deltas:
                for r in rows:
                    if (r.get("scenario") == "H0" and r["temporal"] == temporal
                            and abs(r["delta"] - d) < 1e-9):
                        xs.append(d)
                        ys.append(r[f"type1_error[{key}]"])
                        es.append(r[f"se[{key}]"])
            if not xs:
                continue
            lab = label if temporal == temporals[0] else None
            ax.errorbar(xs, ys, yerr=es, marker=marker, ls=ls, color=color,
                        ms=3.6, lw=1.2, capsize=2.2, alpha=0.9, label=lab)
    ax.axhline(alpha, color="k", ls=":", lw=1.1)
    ax.set_xlabel("composition imbalance  $\\delta$")
    ax.set_title("(a) Type-I error under H0,\nno spatial dependence", fontsize=8.8)
    ax.set_ylim(-0.03, 1.05)

    # ---- (b) power against clustering scale sep (or fall back to the H1 delta curve) ----
    ax = axes[1]
    if power is not None:
        prows = power["rows"]
        pdel = sorted({r["delta"] for r in prows})
        ptemps = [t for t in ("exponential", "inverse_dt", "none")
                  if t in {r["temporal"] for r in prows}]
        seps = sorted({r["sep_m"] for r in prows}, reverse=True)
        for temporal, dls in zip(ptemps, ("-", "--", "-.")):
            for key, (marker, color, label) in styles.items():
                if f"power[{key}]" not in prows[0]:
                    continue
                for delta, ls in zip(pdel, ("-", "--")):
                    xs, ys, es = [], [], []
                    for s in seps:
                        for r in prows:
                            if (r["temporal"] == temporal
                                    and abs(r["sep_m"] - s) < 1e-9
                                    and abs(r["delta"] - delta) < 1e-9):
                                xs.append(s)
                                ys.append(r[f"power[{key}]"])
                                es.append(r[f"se[{key}]"])
                    if not xs:
                        continue
                    lab = (label if (temporal == ptemps[0]
                                     and abs(delta - pdel[0]) < 1e-9) else None)
                    ax.errorbar(xs, ys, yerr=es, marker=marker, ls=ls, color=color,
                                ms=3.4, lw=1.15, capsize=2.0, alpha=0.9, label=lab)
        ax.axhline(alpha, color="k", ls=":", lw=1.1)
        ax.set_xscale("log")
        # default minor tick labels on a log axis overlap, so use the measured sep values
        ax.set_xticks(seps)
        ax.set_xticklabels([f"{s:.0f}" for s in seps])
        ax.minorticks_off()
        ax.set_xlabel("clustering scale $\\mathrm{sep}$ (m), smaller = stronger")
        ax.set_title("(b) Power under within-period\nclustering (H1)",
                     fontsize=8.8)
        # the **shared legend** below explains the style keys (global / block / normal),
        # the subtitle explains the line style (delta), so no duplicate in-axes legend
        # is drawn here. The strong/weak alternative direction is already in the x-axis
        # label; grey text here would collide with the falling curves.
    else:
        for temporal, ls in zip(temporals, ("-", "--")):
            for key, (marker, color, label) in styles.items():
                xs, ys, es = [], [], []
                for d in deltas:
                    for r in rows:
                        if (r.get("scenario") == "H1" and r["temporal"] == temporal
                                and abs(r["delta"] - d) < 1e-9):
                            xs.append(d)
                            ys.append(r[f"type1_error[{key}]"])
                            es.append(r[f"se[{key}]"])
                if not xs:
                    continue
                lab = label if temporal == temporals[0] else None
                ax.errorbar(xs, ys, yerr=es, marker=marker, ls=ls, color=color,
                            ms=3.6, lw=1.2, capsize=2.2, alpha=0.9, label=lab)
        ax.axhline(alpha, color="k", ls=":", lw=1.1)
        ax.set_xlabel("composition imbalance  $\\delta$")
        ax.set_title("(b) Power under within-period\nclustering (H1)", fontsize=8.8)

    axes[0].set_ylabel("rejection rate at $\\alpha=0.05$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    if power is not None:
        sub = ("(a) solid: $w_t = \\exp(-|\\Delta t|/\\tau)$, dashed: $w_t = $ none;"
               "   (b) $w_t = \\exp(-|\\Delta t|/\\tau)$, "
               "solid: $\\delta = 0$, dashed: $\\delta = 0.6$")
    else:
        sub = ("solid: $w_t = \\exp(-|\\Delta t|/\\tau)$;  "
               "dashed: $w_t = $ none")
    fig.text(0.5, 1.02, sub, ha="center", fontsize=7.0, color="#555")
    fig.tight_layout()
    save(fig, "fig3_permutation_validity")


# ---------------------------------------------------------------------------
# Fig.1 study-area point pattern plus the STCLQ matrix
# ---------------------------------------------------------------------------
# Bounding box of the Avon and Somerset force area (lon/lat). All 139,001 records
# belong to that force (checked one by one in the ``Falls within`` field); the box
# frames the map so that the 120 records (0.09%) geocoded outside the force area drop
# out. They affect the extent of the map only, not the statistics.
BOX = (-3.75, -2.05, 50.75, 51.72)
PLACES = [("Bristol", -2.5879, 51.4545, (-33, -10)),
          ("Bath", -2.3590, 51.3811, (4, 2)),
          ("Weston-super-Mare", -2.9770, 51.3460, (4, 2)),
          ("Taunton", -3.1000, 51.0150, (4, 2)),
          ("Yeovil", -2.6330, 50.9420, (4, 2))]
CRIME_CSV = ROOT / "data" / "avon_and_somerset_2023.csv"


def load_crime_points():
    """Read ``data/avon_and_somerset_2023.csv`` into (lon, lat) arrays."""
    lon, lat = [], []
    with open(CRIME_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                lo, la = float(r["longitude"]), float(r["latitude"])
            except (TypeError, ValueError):
                continue
            lon.append(lo)
            lat.append(la)
    return np.asarray(lon), np.asarray(lat)


def fig1_study_area():
    """Fig.1: (a) study-area point pattern and neighbourhood scale; (b) full-period
    STCLQ matrix.

    The figure size follows the GA full-page limit (4.5 in × 7 in), with the two panels
    stacked vertically; the inset in (a) shows the true scale of the two neighbourhood
    radii ``b`` and ``c\\,b``, which is impossible to see on a map of the whole force area.
    """
    lon, lat = load_crime_points()
    inside = (lon > BOX[0]) & (lon < BOX[1]) & (lat > BOX[2]) & (lat < BOX[3])

    R = HERE / "realdata"
    M = np.load(R / "avon_and_somerset_global_matrix.npy")
    av = next(d for d in load_json(R / "summary_all.json")
              if d["dataset"] == "avon_and_somerset")
    names = av["type_names"]
    b = float(av["bandwidth_m"])
    cut = 3.0 * b
    rho, kbar = float(av["rho"]), float(av["kbar"])
    sig = set()
    with open(R / "avon_and_somerset_pairs_global_null.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["significant"].strip().lower() == "true":
                sig.add((r["type_a"], r["type_b"]))

    # local equidistant cylindrical projection (km): scale bar and radii in metres
    lat0 = 0.5 * (BOX[2] + BOX[3])
    kx = np.cos(np.radians(lat0))
    x = (lon - BOX[0]) * 111.320 * kx
    y = (lat - BOX[2]) * 110.574
    W = (BOX[1] - BOX[0]) * 111.320 * kx
    H = (BOX[3] - BOX[2]) * 110.574

    fig = plt.figure(figsize=(4.5, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.0], hspace=0.46,
                          left=0.155, right=0.945, top=0.945, bottom=0.075)

    # ---------------- (a) point pattern ----------------
    ax = fig.add_subplot(gs[0])
    ax.scatter(x[inside], y[inside], s=0.18, c="#2c3e50", alpha=0.15,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xlabel("easting (km)", fontsize=7.6)
    ax.set_ylabel("northing (km)", fontsize=7.6)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.16, linewidth=0.4)
    ax.set_title("(a) Point pattern, Avon and Somerset Constabulary",
                 fontsize=8.4, pad=5)

    for nm, plo, pla, off in PLACES:
        px, py = (plo - BOX[0]) * 111.320 * kx, (pla - BOX[2]) * 110.574
        ax.plot([px], [py], marker="+", ms=3.6, mew=1.0, color="#b03a2e",
                zorder=6)
        ax.annotate(nm, (px, py), textcoords="offset points", xytext=off,
                    fontsize=6.2, color="#7b241c", zorder=6)

    sb = 20.0                                       # 20 km scale bar
    ax.plot([7, 7 + sb], [5.5, 5.5], color="k", lw=1.5, solid_capstyle="butt",
            zorder=5)
    ax.text(7 + sb / 2, 6.9, "20 km", ha="center", fontsize=6.8, zorder=5)
    ax.annotate("", xy=(W - 7, H - 22), xytext=(W - 7, H - 8),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
    ax.text(W - 7, H - 5.5, "N", ha="center", fontsize=7.0)
    ax.text(0.985, 0.028,
            f"$N$ = {lon.size:,} offences over 12 months\n"
            f"{int(inside.sum()):,} inside the force area",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            color="#333333")

    # inset centre: where the most records share one anonymised coordinate
    zlo, zla = -2.5920, 51.4570
    zx, zy = (zlo - BOX[0]) * 111.320 * kx, (zla - BOX[2]) * 110.574
    ax.plot([zx], [zy], marker="s", ms=2.8, color="#b03a2e", zorder=7)
    ax.annotate("", xy=(zx, zy), xycoords="data", xytext=(0.412, 0.76),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="#b03a2e", lw=0.7,
                                shrinkA=0, shrinkB=3))

    # inset: aggregate by unique coordinate, marker size grows with the number of offences
    # that share that anonymised coordinate
    axi = ax.inset_axes([0.045, 0.515, 0.36, 0.457])
    mx = (lon - zlo) * 111320.0 * kx
    my = (lat - zla) * 110574.0
    half = 300.0
    win = (np.abs(mx) <= half) & (np.abs(my) <= half)
    keys, cnt = np.unique(np.stack([np.round(mx[win]), np.round(my[win])], 1),
                          axis=0, return_counts=True)
    axi.scatter(keys[:, 0], keys[:, 1], s=2.0 + 3.2 * np.sqrt(cnt),
                c="#2c3e50", alpha=0.55, linewidths=0)
    for rr, ls, lab, ty_ in ((cut, (0, (3, 2)), f"$c\\,b$ = {cut:.0f} m", 0.90),
                             (b, "-", f"$b$ = {b:.1f} m", 0.56)):
        axi.add_patch(plt.Circle((0, 0), rr, fill=False, ec="#b03a2e", lw=1.0,
                                 ls=ls, zorder=4))
        axi.annotate(lab, xy=(rr / 1.4142, rr / 1.4142),
                     xycoords="data", xytext=(0.965, ty_),
                     textcoords="axes fraction", fontsize=5.9, color="#7b241c",
                     va="center", ha="right",
                     arrowprops=dict(arrowstyle="-", color="#b03a2e", lw=0.6))
    axi.set_xlim(-half, half)
    axi.set_ylim(-half, half)
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_linewidth(0.7)
    ax.text(0.045, 0.497, "zoom: 600 m across", transform=ax.transAxes,
            fontsize=6.0, color="#7b241c", ha="left", va="top")

    # ---------------- (b) STCLQ matrix ----------------
    ax2 = fig.add_subplot(gs[1])
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.15, float(M.min())),
                        vmax=max(1.60, float(M.max())))
    im = ax2.imshow(M, cmap="RdBu_r", norm=norm)
    ax2.set_xticks(range(len(names)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=26, ha="right", fontsize=6.9)
    ax2.set_yticklabels(names, fontsize=6.9)
    ax2.tick_params(length=2)
    ax2.set_xlabel("target type $b$", fontsize=7.6)
    ax2.set_ylabel("source type $a$", fontsize=7.6)
    ax2.set_title(f"(b) STCLQ matrix ($\\rho$ = {rho:.4f}, $\\bar k$ = {kbar:.1f})",
                  fontsize=8.4, pad=5)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                ax2.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc="0.86",
                                        ec="none", zorder=1.5))
                ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.4, color="0.45", zorder=3)
                continue
            star = "*" if (names[i], names[j]) in sig else ""
            ax2.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center",
                     fontsize=6.6, zorder=3)
            if star:
                ax2.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                        ec="k", lw=1.2, zorder=2.5))
    cb = fig.colorbar(im, ax=ax2, fraction=0.043, pad=0.03)
    cb.ax.tick_params(labelsize=6.2)
    cb.outline.set_linewidth(0.6)
    cb.set_label("STCLQ", fontsize=7.0)

    save(fig, "fig1_study_area_stclq")


def main():
    bench_p = HERE / "benchmark" / "benchmark.json"
    par_p = HERE / "parallel" / "parallel.json"
    cal_p = HERE / "calibration" / "calibration.json"

    print("Fig.1 study-area point pattern and STCLQ matrix ...")
    fig1_study_area()

    print("Fig.2 data flow ...")
    fig2_dataflow()

    if bench_p.exists():
        bench = load_json(bench_p)
        print("Fig.4 permutation time ..."); fig4_time(bench)
        print("Fig.5 speed-up ..."); fig5_speedup(bench)
        print("Fig.6 memory and criterion ...")
        table = fig6_memory(bench)
        with open(FIG / "feasible_scale_table.json", "w", encoding="utf-8") as fh:
            json.dump(table, fh, ensure_ascii=False, indent=2)
        print(f"  feasible-scale table -> {FIG / 'feasible_scale_table.json'}")
    else:
        print("[skip] benchmark.json not found")

    if par_p.exists():
        print("Fig.7 parallel scaling ...")
        fig7_parallel(load_json(par_p))
    else:
        print("[skip] parallel.json not found")

    if cal_p.exists():
        print("Fig.3 permutation validity ...")
        fig3_calibration(load_calibration())
    else:
        print("[skip] calibration.json not found")

    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
