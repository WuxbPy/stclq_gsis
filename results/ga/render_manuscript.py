"""Render the real numbers from the experiment output into the manuscript template.

Reads JSON/CSV under ``results/ga/**`` and the Markdown tables under ``tables/``, then
substitutes the ``{{TOKEN}}`` placeholders in ``GA_manuscript.template.md`` in the
submission package directory (resolved by :data:`PAPER` from candidate directory names,
see below).

    PYTHONPATH=src python results/ga/render_manuscript.py
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import re
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Submission package directory: resolved by candidate directory name. Renaming the
# target journal (or the package directory) only needs this line edited.
PAPER = next(
    (p for p in (ROOT / "manuscript",
                 ROOT / "papers" / "Geographical Analysis")
     if p.is_dir()),
    ROOT / "manuscript",
)
TAB = PAPER / "tables"

sys.path.insert(0, str(ROOT / "src"))


def jload(p, default=None):
    p = pathlib.Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def read_text(p, default=""):
    p = pathlib.Path(p)
    return p.read_text(encoding="utf-8") if p.exists() else default


def read_pairs(p):
    """Read ``*_pairs_*.csv`` and return the significant off-diagonal pairs, highest
    stclq first."""
    import csv
    p = pathlib.Path(p)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("significant", "")).strip().lower() != "true":
                continue
            rows.append({
                "a": r["type_a"], "b": r["type_b"],
                "clq": float(r["stclq"]), "p": float(r["p_value"]),
            })
    rows.sort(key=lambda d: -d["clq"])
    return rows


def read_all_pairs(p):
    """Read ``*_pairs_global_null.csv`` and return **all** ordered off-diagonal pairs,
    the non-significant ones included."""
    import csv
    p = pathlib.Path(p)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({"a": r["type_a"], "b": r["type_b"], "clq": float(r["stclq"]),
                        "sig": str(r.get("significant", "")).strip().lower() == "true"})
    return out


def cpu_info():
    model, cores = "unknown CPU", 1
    try:
        txt = pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        cores = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip() or "1"
    except Exception:
        pass
    return model, int(cores)


def ram_gb():
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(round(int(line.split()[1]) / 1024 / 1024))
    except Exception:
        pass
    return 0


def fmt_int(x):
    return f"{int(round(x)):,}"


def fmt_p(x):
    """Render a p-value in readable form (three significant digits)."""
    if x is None:
        return "?"
    x = float(x)
    if x == 0:
        return "0"
    if x >= 0.01:
        return f"{x:.3f}"
    return f"{x:.3g}"


def _pair_list(rs):
    """["A–B", "C–D"] -> "A–B and C–D" (English conjunction style)."""
    names = [f"{r['type_a']}–{r['type_b']}" for r in rs]
    if not names:
        return "*(none)*"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _stab(T, prefix, stab):
    """Expand the month-to-month stability JSON into template tokens (§6.6)."""
    n = stab["n_periods"]
    T[f"{prefix}_N_MONTHS"] = str(n)
    T[f"{prefix}_N_PAIRS"] = str(stab["n_pairs"])
    T[f"{prefix}_N_STABLE"] = str(stab["n_pairs_stable"])
    T[f"{prefix}_STABLE_PAIRS"] = _pair_list([r for r in stab["pairs"] if r["sign_consistent"]])
    top = max(stab["pairs"], key=lambda r: r["pooled_stclq"])
    T[f"{prefix}_TOP_PAIR"] = f"{top['type_a']}–{top['type_b']}"
    T[f"{prefix}_TOP_MIN"] = f"{top['min']:.2f}"
    T[f"{prefix}_TOP_MAX"] = f"{top['max']:.2f}"
    T[f"{prefix}_TOP_MED"] = f"{top['median']:.2f}"
    T[f"{prefix}_TOP_MONTHS"] = str(top["n_months_attract"])
    partial = sorted([r for r in stab["pairs"] if r["n_months_attract"] < n],
                     key=lambda r: r["n_months_attract"])
    T[f"{prefix}_PARTIAL_N"] = str(len(partial))
    T[f"{prefix}_PARTIAL_LIST"] = ("; ".join(
        f"{r['type_a']}–{r['type_b']} ({r['n_months_attract']} of {n})" for r in partial)
        or "*(none)*")
    T[f"{prefix}_MONTHS_MIN"] = str(stab["min_months_attract"])
    T[f"{prefix}_MONTHS_MAX"] = str(stab["max_months_attract"])


def _top(T, prefix, pairs):
    """Write the strongest significant pair into ``<prefix>_PAIR`` / ``<prefix>_CLQ``."""
    if pairs:
        d = pairs[0]
        T[f"{prefix}_PAIR"] = f"{d['a']}–{d['b']}"
        T[f"{prefix}_CLQ"] = f"{d['clq']:.2f}"
    else:
        T[f"{prefix}_PAIR"] = "*(none)*"
        T[f"{prefix}_CLQ"] = "--"


def build_tokens():
    T = {}
    import numpy as np  # noqa: F401
    import scipy

    # Experiment environment: **read the frozen provenance.json first** (the host and
    # software versions the experiment actually ran on) and fall back to probing this
    # machine only when it is missing. Otherwise a re-render elsewhere would replace the
    # host with the rendering machine, misreporting the environment, which is far worse
    # than a typesetting error.
    prov = jload(HERE / "provenance.json") or {}
    model, cores = cpu_info()
    T["CPU_MODEL"] = str(prov.get("cpu_model", model))
    T["CPU_CORES"] = str(prov.get("cpu_cores", cores))
    T["RAM_GB"] = str(prov.get("ram_gb", ram_gb()))
    T["PY_VERSION"] = str(prov.get("python", platform.python_version()))
    T["NUMPY_VERSION"] = str(prov.get("numpy", np.__version__))
    T["SCIPY_VERSION"] = str(prov.get("scipy", scipy.__version__))

    # ---- correctness ----
    corr = jload(HERE / "correctness" / "correctness.json")
    if corr:
        s = corr["summary"]
        T["N_CORRECTNESS"] = str(s["n_cases"])
        T["MAX_ABS_DIFF"] = f"{s['max_abs_diff']:.2e}"
        T["MAX_ABS_DIFF_COLLAPSED"] = f"{s['max_abs_diff']:.1e}"
        # Number of cases whose deviation is **exactly 0**: only these are "bit-for-bit
        # identical"; the rest sit at floating-point rounding of order 1e-15 to 1e-14.
        T["N_EXACT"] = str(sum(1 for c in corr["cases"]
                               if float(c["max_abs_diff"]) == 0.0))
        # Number of cases at the global and at the local level (the factorial grid)
        T["N_GLOBAL"] = str(sum(1 for c in corr["cases"] if c["level"] == "global"))
        T["N_LOCAL"] = str(sum(1 for c in corr["cases"] if c["level"] == "local"))

    # ---- benchmark ----
    bench = jload(HERE / "benchmark" / "benchmark.json")
    if bench:
        rows, fits = bench["rows"], bench["fits"]
        T["PERM_SLOPE"] = f"{fits['perm_slope']:.2f}"
        T["PERM_R2"] = f"$R^2$ = {fits['perm_r2']:.4f}"
        T["REF_SLOPE"] = f"{fits['ref_slope']:.2f}"
        T["REF_R2"] = f"$R^2$ = {fits['ref_r2']:.4f}"
        sp = [(r["n"], r["speedup_vs_ref"]) for r in rows
              if np.isfinite(r["speedup_vs_ref"])]
        if sp:
            nmin, smin = sp[0]
            nmax, smax = sp[-1]
            T["SPEEDUP_MIN"] = f"{smin:,.0f}×"
            T["SPEEDUP_MAX"] = f"{smax:,.0f}×"
            T["SPEEDUP_MIN_N"] = fmt_int(nmin)
            T["SPEEDUP_MAX_N"] = fmt_int(nmax)
        last = rows[-1]
        T["N_MAX"] = fmt_int(last["n"])
        T["NNZ_MAX"] = f"{last['nnz']:.3e}".replace("e+0", "×10^").replace(
            "e+", "×10^") if False else f"{last['nnz'] / 1e6:,.1f} M"
        T["PERM_MS_MAX"] = f"{last['perm_s']*1e3:,.1f}"
        T["T999_MAX"] = f"{last['time_999_perms_s']:,.1f}"
        T["RSS_MAX"] = fmt_int(last["rss_mb"])
        T["BUILD_MAX"] = f"{last['build_s']:,.1f}"
        T["DENSE_GB_MAX"] = f"{last['dense_mb_est']/1024:,.0f}"
        T["REF_N_MAX"] = fmt_int(bench["settings"]["max_ref_n"])
        # ratio of the two fitted power laws → speed-up extrapolated to N=10^6
        if np.isfinite(fits["ref_slope"]) and np.isfinite(fits["perm_slope"]) and sp:
            # the slope is the difference of the two **cost-law** fits (both R² > 0.99),
            # more stable than fitting the noisier speed-up series directly; the anchor is
            # the measured speed-up at the largest N (the last point that still has a
            # reference measurement). Fig.5 (make_figures.fig5_speedup) must use the same
            # anchor, otherwise figure and text disagree.
            s = fits["ref_slope"] - fits["perm_slope"]
            c = sp[-1][1] / sp[-1][0] ** s
            T["SPEEDUP_EXTRAP_1E6"] = f"{c * 1e6 ** s:,.0f}×"
        # fit a power law to the measured reference times and extrapolate to N = 139,001
        rr = [(r["n"], r["ref_perm_s"]) for r in rows
              if np.isfinite(r["ref_perm_s"])]
        if len(rr) >= 2:
            ln = np.log([x for x, _ in rr])
            lt = np.log([y for _, y in rr])
            s, a = np.polyfit(ln, lt, 1)
            t139 = float(np.exp(a + s * np.log(139_001.0)))
            T["REF_S_PERM_139K"] = f"{t139:,.0f}"
            T["REF_TOTAL_139K"] = f"{t139 * 999 / 3600:,.0f} h"

    # ---- feasible scale ----
    feas = jload(HERE / "figures" / "feasible_scale_table.json")
    if feas:
        T["B_MAX"] = f"{feas[0]['b_max_rho']:.0f}"
        for r in feas:
            if r["n"] == 1_000_000:
                T["NNZ_1E6"] = f"{r['nnz_at_b_max']/1e6:,.0f} M"
                T["CSR_GB_1E6"] = f"{r['csr_mb_at_b_max']/1024:,.1f}"

    # ---- parallel ----
    par = jload(HERE / "parallel" / "parallel.json")
    if par:
        T["PAR_N"] = fmt_int(par["settings"]["n"])
        T["PAR_NPERM"] = str(par["settings"]["n_perm"])
        # The strong-scaling curve is **non-monotonic** (it falls away past the CPU-bound
        # region), so the "peak" must be the p with the largest speedup on the curve and
        # **not** the last p in the sweep.
        strong = par.get("strong") or []
        if strong:
            peak = max(strong, key=lambda s: s["speedup"])
            last = strong[-1]
            T["PAR_SPEEDUP_MAX"] = f"{peak['speedup']:.1f}"
            T["PAR_PMAX"] = str(peak["p"])
            T["PAR_SPEEDUP_LAST"] = f"{last['speedup']:.1f}"
            T["PAR_PLAST"] = str(last["p"])
        plat = par.get("platform") or {}
        if plat.get("max_cpu_speedup"):
            T["PLAT_CPU_MAX"] = f"{plat['max_cpu_speedup']:.1f}"
            best = max(plat["cpu"], key=lambda c: c["speedup"])
            T["PLAT_CPU_MAX_P"] = str(best["p"])
            worst = plat["cpu"][-1]
            T["PLAT_CPU_LAST"] = f"{worst['speedup']:.1f}"
            T["PLAT_CPU_LAST_P"] = str(worst["p"])
            if strong:
                T["PAR_VS_CPU"] = \
                    f"{100.0 * max(s['speedup'] for s in strong) / plat['max_cpu_speedup']:.0f}"
        if plat.get("bandwidth"):
            T["PLAT_BW_1"] = f"{plat['bandwidth'][0]['per_process_gbs']:.0f}"
            T["PLAT_BW_MAX"] = f"{plat['peak_aggregate_gbs']:.0f}"
            T["PLAT_BW_REL"] = f"{plat['max_bw_relative']:.1f}"
            # engine peak as a fraction of the platform ceiling: this separates the
            # efficiency of the algorithm from the bandwidth limit of the host
            if strong and plat["peak_aggregate_gbs"] > 0:
                best_bw = plat["bandwidth"][0]["per_process_gbs"]
                ceil_sp = plat["peak_aggregate_gbs"] / best_bw if best_bw else 0.0
                if ceil_sp > 0:
                    T["PAR_FRAC_OF_BW"] = \
                        f"{100.0 * max(s['speedup'] for s in strong) / ceil_sp:.0f}"
        w = par.get("weak") or []
        if w:
            T["PAR_WEAK_EFF_MIN"] = f"{min(x['efficiency'] for x in w):.2f}"
            T["PAR_PREFIX_OK"] = str(all(x.get("prefix_identical", True)
                                         for x in w))
        ctrl = (plat.get("weak_control") or [])
        if not ctrl and plat.get("bandwidth"):
            bw1 = plat["bandwidth"][0]["per_process_gbs"]
            ctrl = [{"p": b["p"], "efficiency": b["per_process_gbs"] / bw1}
                    for b in plat["bandwidth"] if b["per_process_gbs"] > 0]
        if ctrl:
            T["PAR_CTRL_EFF_MIN"] = f"{min(c['efficiency'] for c in ctrl):.2f}"
        ov = par.get("startup_overhead") or []
        if ov:
            small, large = ov[0], ov[-1]
            T["PAR_SMALL_N"] = fmt_int(small["n"])
            T["PAR_SMALL_CSR"] = f"{small['csr_mb']:.0f}"
            T["PAR_SMALL_SPEEDUP"] = f"{small['speedup']:.2f}"
            T["PAR_LARGE_N"] = fmt_int(large["n"])
            T["PAR_LARGE_CSR"] = f"{large['csr_mb']:.0f}"
            T["PAR_LARGE_SPEEDUP"] = f"{large['speedup']:.2f}"
            T["PAR_SCAN_SP_MIN"] = f"{min(x['speedup'] for x in ov):.2f}"
            T["PAR_SCAN_SP_MAX"] = f"{max(x['speedup'] for x in ov):.2f}"

    # ---- host state + configuration-regime comparison (exp_regime) ----
    reg = jload(HERE / "regime" / "regime.json")
    pat = jload(HERE / "regime" / "pattern.json")
    if pat:
        r, s = pat["real"], pat["synthetic"]
        T["PAT_REAL_NNZ"] = fmt_int(r["nnz"])
        T["PAT_SYN_NNZ"] = fmt_int(s["nnz"])
        T["PAT_NNZ_DIFF"] = f"{pat['nnz_rel_diff_pct']:.2f}"
        T["PAT_REAL_CSR"] = f"{r['csr_mb']:.0f}"
        T["PAT_SYN_CSR"] = f"{s['csr_mb']:.0f}"
        T["PAT_REAL_PERM"] = f"{r['perm_ms']:.0f}"
        T["PAT_SYN_PERM"] = f"{s['perm_ms']:.0f}"
        T["PAT_RATIO"] = f"{pat['ratio_synth_over_real']:.2f}"
    if reg:
        h = (reg.get("host") or [{}])[0]
        if h.get("cpu_model"):
            T["HOST_CPU_MODEL"] = h["cpu_model"]
        T["HOST_CPUS"] = str(h.get("nproc", "?"))
        T["HOST_CPUS_ALLOWED"] = str(h.get("cpus_allowed", "?"))
        T["HOST_MEMS"] = str(h.get("mems_allowed", "?"))
        T["HOST_THREADS"] = str(h.get("n_threads", "?"))
        T["HOST_THROTTLED"] = str(h.get("cpu_nr_throttled", "?"))
        topo = h.get("topology") or {}
        if topo:
            T["HOST_P_THREADS"] = str(topo.get("n_p_threads", "?"))
            T["HOST_E_THREADS"] = str(topo.get("n_e_threads", "?"))
            if topo.get("max_freq_khz"):
                T["HOST_P_GHZ"] = f"{topo['max_freq_khz'] / 1e6:.1f}"
        loads = [x["loadavg"][0] for x in (reg.get("host") or [])
                 if x.get("loadavg")]
        if loads:
            T["HOST_LOAD_MIN"] = f"{min(loads):.1f}"
            T["HOST_LOAD_MAX"] = f"{max(loads):.1f}"
        rows = reg.get("rows") or []
        if rows:
            # speed-up within one round for the §6.4 control configuration and the
            # §6.6 Avon run
            by = {r["name"]: r for r in rows}
            sw = by.get("sweep_K8_N100k")
            av = by.get("avon_K4_N139k")
            if sw:
                T["REG_SWEEP_SPEEDUP"] = f"{sw['speedup']:.2f}"
                T["REG_SWEEP_GBS"] = f"{sw['gbs_parallel']:.1f}"
                T["REG_SWEEP_CSR"] = f"{sw['csr_mb']:.0f}"
            if av:
                T["REG_AVON_SPEEDUP"] = f"{av['speedup']:.2f}"
                T["REG_AVON_GBS"] = f"{av['gbs_parallel']:.1f}"
                T["REG_AVON_CSR"] = f"{av['csr_mb']:.0f}"
            if sw and av:
                T["REG_RATIO"] = f"{av['speedup'] / sw['speedup']:.1f}"
            # upper bound for K=4 among the DRAM-resident configurations (K=4 beats
            # K=8 at both N)
            k4 = [r for r in rows if r["K"] == 4 and r["csr_mb"] > 100]
            if k4:
                T["REG_K4_SPEEDUP"] = f"{max(r['speedup'] for r in k4):.2f}"
            csv = by.get("city_K4_N16k")
            if csv:
                T["REG_CITY_SPEEDUP"] = f"{csv['speedup']:.2f}"
                T["REG_CITY_GBS"] = f"{csv['gbs_parallel']:.1f}"
                T["REG_CITY_CSR"] = f"{csv['csr_mb']:.0f}"

    # ---- calibration ----
    cal = jload(HERE / "calibration" / "calibration.json")
    if cal:
        rows = cal["rows"]
        T["CAL_REPS"] = str(cal["settings"]["reps"])
        h0 = [r for r in rows if r["scenario"] == "H0"
              and r["temporal"] == "exponential"]
        if h0:
            worst = max(h0, key=lambda r: r["delta"])
            T["CAL_GLOBAL_ERR"] = f"{worst['type1_error[perm:type]']:.2f}"
            T["CAL_BLOCK_ERR"] = f"{worst['type1_error[perm:type_period]']:.2f}"
            T["CAL_NORM_ERR"] = f"{worst['type1_error[norm:type]']:.2f}"
            T["CAL_DELTA_MAX"] = f"{worst['delta']:.1f}"
            mid = [r for r in h0 if abs(r["delta"] - 0.3) < 1e-9]
            if mid:
                T["CAL_GLOBAL_ERR_D3"] = f"{mid[0]['type1_error[perm:type]']:.2f}"
                T["CAL_BLOCK_ERR_D3"] = \
                    f"{mid[0]['type1_error[perm:type_period]']:.2f}"
        # The point-estimate bias is taken from the **H0 arm**: there type and location
        # are independent within a period, so "type ⊥ space" holds, and the drift of the
        # observed CLQ(1,1) with delta once the temporal kernel is switched on is pure
        # bias in the statistic. (The H1 arm builds in spatial clustering, so its rising
        # observed value is genuine power and cannot support a bias argument.)
        if h0:
            worst_obs = max(h0, key=lambda r: r["delta"])
            T["CAL_OBS_BIAS"] = f"{worst_obs['mean_observed_pair']:.2f}"

    # ---- power sweep ----
    pw = jload(HERE / "calibration" / "calibration_power.json")
    if pw:
        prows = [r for r in pw["rows"] if r["temporal"] == "exponential"]
        T["POW_REPS"] = str(pw["settings"]["reps"])
        d0 = [r for r in prows if abs(r["delta"]) < 1e-9]
        if d0:
            gaps = [abs(r["power[perm:type]"] - r["power[perm:type_period]"])
                    for r in d0]
            T["POW_GAP_D0"] = f"{max(gaps):.2f}"
            strong = min(d0, key=lambda r: r["sep_m"])
            T["POW_STRONG_GLOBAL"] = f"{strong['power[perm:type]']:.2f}"
            T["POW_STRONG_BLOCK"] = f"{strong['power[perm:type_period]']:.2f}"
            T["POW_STRONG_SEP"] = f"{strong['sep_m']:.0f}"
            weak = max(d0, key=lambda r: r["sep_m"])
            T["POW_WEAK_GLOBAL"] = f"{weak['power[perm:type]']:.2f}"
            T["POW_WEAK_BLOCK"] = f"{weak['power[perm:type_period]']:.2f}"
        d6 = [r for r in prows if abs(r["delta"] - 0.6) < 1e-9]
        if d6:
            weak = max(d6, key=lambda r: r["sep_m"])
            T["POW_GLOBAL_D6"] = f"{weak['power[perm:type]']:.2f}"
            T["POW_BLOCK_D6"] = f"{weak['power[perm:type_period]']:.2f}"
            T["POW_GLOBAL_D6_SEP"] = f"{weak['sep_m']:.0f}"

    # ---- real data ----
    real = jload(HERE / "realdata" / "summary_all.json")
    if real:
        a = real[0]
        T["REAL_N"] = fmt_int(a["n_points"])
        T["REAL_TYPES"] = str(a["n_types"])
        T["REAL_PERIODS"] = str(a["n_periods"])
        T["REAL_BUILD"] = f"{a['build_s']:.1f}"
        T["REAL_NNZ"] = f"{a['nnz']:,}"
        T["REAL_NPERM"] = str(a["n_perm"])
        T["REAL_NJOBS"] = str(a["n_jobs"])
        T["REAL_SERIAL"] = f"{a['serial_seconds']:.1f}"
        T["REAL_PARALLEL"] = f"{a['parallel_seconds']:.1f}"
        T["REAL_SPEEDUP"] = f"{a['parallel_speedup']:.1f}"
        T["REAL_SIG_GLOBAL"] = str(a["n_significant_fdr[type]"])
        T["REAL_SIG_BLOCK"] = str(a["n_significant_fdr[type_period]"])
        T["REAL_PAIRS"] = str(a.get("n_pairs_offdiag", a["n_types"] * (a["n_types"] - 1)))
        T["REAL_FLIP"] = str(a.get("n_decision_flip", "?"))
        T["REAL_MAXPDIFF"] = fmt_p(a.get("max_abs_p_diff"))
        T["REAL_P_INCR"] = str(a.get("n_p_increase", "?"))
        T["REAL_P_DECR"] = str(a.get("n_p_decrease", "?"))
        _top(T, "REAL_TOP", read_pairs(HERE / "realdata" /
                                       f"{a['dataset']}_pairs_global_null.csv"))
        # all significant pairs (STCLQ descending) and the weakest ordered pair, for the
        # item-by-item account in §6.6
        sig = read_pairs(HERE / "realdata" / f"{a['dataset']}_pairs_global_null.csv")
        names = [f"{d['a']}→{d['b']} ({d['clq']:.2f})" for d in sig]
        T["REAL_SIG_LIST"] = (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1 \
            else (names[0] if names else "*(none)*")
        allp = read_all_pairs(HERE / "realdata" / f"{a['dataset']}_pairs_global_null.csv")
        if allp:
            lo = min(allp, key=lambda d: d["clq"])
            T["REAL_MIN_PAIR"] = f"{lo['a']}→{lo['b']}"
            T["REAL_MIN_CLQ"] = f"{lo['clq']:.2f}"
            rec = next((d for d in allp if d["a"] == lo["b"] and d["b"] == lo["a"]), None)
            if rec:
                T["REAL_MIN_RECIP_PAIR"] = f"{lo['b']}→{lo['a']}"
                T["REAL_MIN_RECIP_CLQ"] = f"{rec['clq']:.2f}"
                T["REAL_MIN_RATIO"] = f"{rec['clq'] / lo['clq']:.1f}"
        if len(real) > 1:
            b = real[1]
            T["REAL2_N"] = fmt_int(b["n_points"])
            T["REAL2_TYPES"] = str(b["n_types"])
            T["REAL2_PERIODS"] = str(b["n_periods"])
            T["REAL2_SIG_GLOBAL"] = str(b["n_significant_fdr[type]"])
            T["REAL2_SIG_BLOCK"] = str(b["n_significant_fdr[type_period]"])
            T["REAL2_PAIRS"] = str(b.get("n_pairs_offdiag",
                                         b["n_types"] * (b["n_types"] - 1)))
            T["REAL2_FLIP"] = str(b.get("n_decision_flip", "?"))
            T["REAL2_MAXPDIFF"] = fmt_p(b.get("max_abs_p_diff"))
            T["REAL2_P_INCR"] = str(b.get("n_p_increase", "?"))
            _top(T, "REAL2_TOP", read_pairs(HERE / "realdata" /
                                            f"{b['dataset']}_pairs_global_null.csv"))

    # ---- month-to-month stability (§6.6 temporal evidence, from exp_stability.py) ----
    for prefix, ds in (("STAB", "avon_and_somerset"), ("STAB2", "city_of_london")):
        p = HERE / "realdata" / f"{ds}_period_stability.json"
        if p.exists():
            _stab(T, prefix, jload(p))

    # ---- tables ----
    for key, fn in (("TABLE1", "table1_algorithm.md"),
                    ("TABLE2", "table2_complexity.md"),
                    ("TABLE3", "table3_benchmark.md")):
        T[key] = read_text(TAB / fn, f"*(Table {key[-1]} not generated yet)*")

    # ---- reference list (generated from references.bib, never hand-written) ----
    # GA (Wiley / Chicago) style: the reference list is alphabetical by author and the
    # text cites author and year, so the list is generated from the bib as a single
    # source, ruling out drift between bib and hand-written list.
    try:
        sys.path.insert(0, str(HERE))
        import make_refs
        entries = make_refs.parse_bib()
        T["REFERENCES"] = make_refs.render(entries)
        dup = make_refs.duplicate_dois(entries)
        if dup:
            print("[warn] references.bib contains duplicate DOIs:")
            for a, b, d in dup:
                print(f"   - {a} and {b} share {d}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] cannot render the reference list: {type(exc).__name__}: {exc}")
        T["REFERENCES"] = "*(reference list not generated)*"
    return T


BACKUP = HERE / "_render_backup"


def safe_write(dst: pathlib.Path, out: str) -> None:
    """Write the rendered manuscript; if the **content differs**, back up the old file to
    ``_render_backup/`` first.

    Why this gate exists
    --------------------
    The rendered file is a **product** and the template is the editing entry point. If
    someone edits the rendered file directly (to fill in affiliations, say), the next
    render silently wipes those changes, and that has already happened once on this
    project. Copying the old file before overwriting, and reporting the manual edit
    explicitly, at least makes the accident recoverable.
    """
    if dst.exists():
        old = dst.read_text(encoding="utf-8")
        if old != out:
            BACKUP.mkdir(exist_ok=True)
            keep = BACKUP / f"{dst.stem}.{time.strftime('%Y%m%d-%H%M%S')}.md"
            keep.write_text(old, encoding="utf-8")
            print(f"  [guard] old render differs from this output -> backed up {keep.relative_to(ROOT)}")
            print("  [guard] if that was a manual edit: the render is not the entry point, merge the change into the template, re-render, then delete the backup.")
    dst.write_text(out, encoding="utf-8")


def main():
    T = build_tokens()
    for tpl_name, out_name in (("GA_manuscript", "GA_manuscript.md"),
                               ("cover_letter", "cover_letter.md")):
        tpl_p = PAPER / f"{tpl_name}.template.md"
        if not tpl_p.exists():
            print(f"[skip] {tpl_p.name}")
            continue
        tpl = tpl_p.read_text(encoding="utf-8")
        missing = sorted({m for m in re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)
                          if m not in T})
        out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: T.get(m.group(1), m.group(0)),
                     tpl)
        dst = PAPER / out_name
        safe_write(dst, out)
        n_tok = len(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)))
        print(f"rendered -> {dst}  ({n_tok - len(missing)}/{n_tok} tokens filled)")
        if missing:
            print("  [warn] unresolved tokens (experiment output missing?):")
            for m in missing:
                print("   -", m)


if __name__ == "__main__":
    main()
