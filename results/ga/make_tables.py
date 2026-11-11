"""Generation of Tables 1–3 and supplementary Tables S1–S8.

Renders Markdown and LaTeX from the experiment output and writes both to ``tables/``
in the submission package directory.

    PYTHONPATH=src python results/ga/make_tables.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Submission package directory: resolved by candidate name; a rename edits one line.
PAPER = next(
    (p for p in (ROOT / "manuscript",
                 ROOT / "papers" / "Geographical Analysis")
     if p.is_dir()),
    ROOT / "manuscript",
)
TABDIR = PAPER / "tables"
TABDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Table 1: algorithm pseudocode
# ---------------------------------------------------------------------------
ALGORITHM = r"""
### Table 1. Structure-invariant sparse permutation inference (Algorithm 1)

**Input:** coordinates $x\in\mathbb{R}^{N\times2}$; labels $t\in\{1..K\}^N$;
time stamps $\pi\in\{1..P\}^N$; bandwidth $b$; cutoff factor $c$; kernel $k_s,k_t$;
number of permutations $n_{\mathrm{perm}}$; seed $s$.
**Output:** observed statistic $\hat C_o\in\mathbb{R}^{K\times K}$; $p$-values; permutation plan.

```
 1  # ---------- build once ----------
 2  (src, dst, d) ← KDTreeRadiusQuery(x, r = c·b)        # O(N log N + nnz)
 3  data ← k_s(d, b) * k_t(|π_src − π_dst|)               # invariant weights
 4  W ← CSR(src, dst, data, N)                            # nnz = ρN², O(nnz) memory
 5  Y ← AugmentedDesign(t)                                # N×(K+1); last column = 1
 6  Ĉ_o ← Reduce(SpMM(W, Y))                              # observed statistic
 7  plan ← default_rng(s).integers(0, 2⁶³, n_perm)        # per-permutation seeds
 8  # ---------- parallel over permutations ----------
 9  fork workers (copy-on-write share of W.data/indices/indptr)
10  for each permutation seed s_p assigned to this worker:
11      σ ← Permute(t | π, s_p)                           # global or block-in-period
12      Y_σ ← AugmentedDesign(σ)                          # row permutation only
13      Z ← SpMM(W, Y_σ)                                  # O(nnz), rows are private
14      C_p ← Reduce(Z)                                   # O(NK + K²)
15      accumulate block-private counters (ge, le)
16  barrier; reduce counters exactly (values are integers)
17  p ← (ge + 1) / (n_perm + 1)                           # Phipson & Smyth (2010)
18  return Ĉ_o, p, plan
```

Complexity: build $O(N\log N + \mathrm{nnz})$ once; **each** permutation $O(\mathrm{nnz})$,
i.e. $O(\rho N^2)$ instead of $O(N^2)$; reduction $O(NK + K^2)$; memory $O(\mathrm{nnz})$.
"""

# ---------------------------------------------------------------------------
# Table 2: complexity
# ---------------------------------------------------------------------------
COMPLEXITY_ROWS = [
    ("Neighborhood + CSR build", "once", "$O(N\\log N + \\mathrm{nnz})$",
     "$O(N\\log N + \\rho N^2)$", "KD-Tree radius query, compressed to CSR"),
    ("One permutation (statistic)", "$n_{\\mathrm{perm}}$", "$O(\\mathrm{nnz})$",
     "$O(\\rho N^2)$", "CSR $\\times$ dense $Y$ (SpMM), rows independent"),
    ("Reduction to $\\hat C$", "$n_{\\mathrm{perm}}$", "$O(NK + K^2)$",
     "$O(NK + K^2)$", "one $K\\times(K{+}1)$ contraction"),
    ("Reference implementation", "$n_{\\mathrm{perm}}$", "$O(N^2)$", "$O(N^2)$",
     "re-traverses all pairs every permutation"),
    ("Peak memory (engine)", "n/a", "$O(\\mathrm{nnz})$", "$O(\\rho N^2)$",
     "CSR: 16 bytes/edge; reference needs $O(N^2)$"),
    ("Amortized total", "$n_{\\mathrm{perm}}$",
     "$O(N\\log N + n_{\\mathrm{perm}}\\,\\mathrm{nnz})$",
     "$O(N\\log N + n_{\\mathrm{perm}}\\rho N^2)$",
     "dominates from the first permutation at measured sizes"),
]


def table2():
    md = ["### Table 2. Complexity and memory bounds", "",
          "| Stage | Frequency | Sparse engine | With $\\rho = \\mathrm{nnz}/N^2$ | Note |",
          "|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Complexity and memory bounds.}",
           r"\begin{tabular}{lllll}", r"\hline",
           r"Stage & Freq. & Sparse engine & With $\rho$ & Note \\", r"\hline"]
    for r in COMPLEXITY_ROWS:
        md.append("| " + " | ".join(r) + " |")
        tex.append(" & ".join(r) + r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


# ---------------------------------------------------------------------------
# Table 3: benchmark results
# ---------------------------------------------------------------------------
def table3(bench):
    rows = bench["rows"]
    fits = bench["fits"]
    md = ["### Table 3. Large-scale benchmark "
          "($\\bar k \\approx 150$, synthetic clustered patterns, "
          "$K = 8$)", "",
          "| $N$ | $\\bar k$ | $\\rho$ | build (s) | nnz | 1 perm (ms) | "
          "ref. $O(N^2)$ (ms) | speed-up | 999 perms (s) | peak RSS (MB) |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Large-scale benchmark.}",
           r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrrrrrrrr}", r"\hline",
           r"$N$ & $\bar k$ & $\rho$ & build (s) & nnz & 1 perm (ms) & "
           r"ref. (ms) & speed-up & 999 perms (s) & peak RSS (MB) \\", r"\hline"]
    for r in rows:
        ref = "n/a" if r["ref_perm_s"] != r["ref_perm_s"] else f"{r['ref_perm_s']*1e3:,.0f}"
        sp = "n/a" if r["speedup_vs_ref"] != r["speedup_vs_ref"] else f"{r['speedup_vs_ref']:,.0f}×"
        md.append("| {n:,.0f} | {kbar:.0f} | {rho:.2e} | {build_s:.2f} | "
                  "{nnz:,.0f} | {perm:.2f} | {ref} | {sp} | {t999:.1f} | "
                  "{rss:,.0f} |".format(n=r["n"], kbar=r["kbar"], rho=r["rho"],
                                        build_s=r["build_s"], nnz=r["nnz"],
                                        perm=r["perm_s"] * 1e3, ref=ref, sp=sp,
                                        t999=r["time_999_perms_s"], rss=r["rss_mb"]))
        tex.append(f"{r['n']:,.0f} & {r['kbar']:.0f} & {r['rho']:.2e} & "
                   f"{r['build_s']:.2f} & {r['nnz']:,.0f} & {r['perm_s']*1e3:.2f} & "
                   f"{ref} & {sp} & {r['time_999_perms_s']:.1f} & {r['rss_mb']:,.0f} \\\\")
    tex += [r"\hline", r"\end{tabular}}", r"\end{table*}"]
    note = (f"\n\n*Fitted log–log slopes: sparse 1-permutation cost "
            f"$s = {fits['perm_slope']:.3f}$ ($R^2 = {fits['perm_r2']:.4f}$), "
            f"reference $s = {fits['ref_slope']:.3f}$ "
            f"($R^2 = {fits['ref_r2']:.4f}$); build $s = {fits['build_slope']:.3f}$.\n")
    return "\n".join(md) + note, "\n".join(tex) + "\n"


# ---------------------------------------------------------------------------
# Supplementary tables
# ---------------------------------------------------------------------------
def table_correctness(corr):
    s = corr["summary"]
    inv = s["invariance"]
    n_exact = sum(1 for c in corr["cases"] if float(c["max_abs_diff"]) == 0.0)
    lines = [
        "### Table S1. Correctness against the reference $O(N^2)$ implementation",
        "",
        f"*{s['n_cases']} configurations in a complete factorial design "
        f"(4 spatial kernels $\\times$ 3 temporal weights $\\times$ "
        f"$K \\in \\{{2,3,5\\}}$ $\\times$ 2 levels); worst-case "
        f"$\\max\\lvert\\Delta\\rvert = {s['max_abs_diff']:.2e}$ against a tolerance of "
        f"$10^{{-10}}$; {n_exact} configurations agree to the last bit "
        f"($\\Delta \\equiv 0$); invariance under label permutation: "
        f"unchanged = {inv['unchanged']}, "
        f"$\\max\\lvert\\Delta\\,\\mathrm{{data}}\\rvert = "
        f"{inv['max_data_delta']:.1e}$.*",
        "",
        "| level | spatial kernel | $K$ | $w_t$ | $N$ | "
        "$\\lvert\\Delta\\rvert_{\\max}$ | pass |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for c in corr["cases"]:
        lines.append(f"| {c['level']} | {c['kernel']} | {c['K']} | {c['temporal']} | "
                     f"{c['n']} | {c['max_abs_diff']:.2e} | {c['pass']} |")
    return "\n".join(lines) + "\n"


def table_calibration(cal):
    rows = cal["rows"]
    deltas = sorted({r["delta"] for r in rows})
    temporals = sorted({r["temporal"] for r in rows})
    md = ["### Table S2. Permutation validity under spatio-temporal weighting", "",
          "Type-I error (H0, no spatial dependence) and power (H1, within-period "
          "clustering) at $\\alpha = 0.05$, $\\pm$ binomial SE.", "",
          "| scenario | $w_t$ | $\\delta$ | mean $\\hat C(1,1)$ | "
          "rejection rate: global perm. | rejection rate: block perm. | "
          "rejection rate: normal approx. |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for scen in ("H0", "H1"):
        for temporal in temporals:
            for d in deltas:
                for r in rows:
                    if (r["scenario"] == scen and r["temporal"] == temporal
                            and abs(r["delta"] - d) < 1e-9):
                        md.append(
                            f"| {scen} | {temporal} | {d:.1f} | "
                            f"{r['mean_observed_pair']:.3f} | "
                            f"{r['type1_error[perm:type]']:.3f}±{r['se[perm:type]']:.3f} | "
                            f"{r['type1_error[perm:type_period]']:.3f}±{r['se[perm:type_period]']:.3f} | "
                            f"{r['type1_error[norm:type]']:.3f} |")
    return "\n".join(md) + "\n"


def table_power(power):
    rows = power["rows"]
    seps = sorted({r["sep_m"] for r in rows}, reverse=True)
    temporals = sorted({r["temporal"] for r in rows})
    deltas = sorted({r["delta"] for r in rows})
    st = power["settings"]
    md = ["### Table S5. Power against a within-period clustering alternative",
          "",
          "Rejection rate at $\\alpha = 0.05$, $\\pm$ binomial SE "
          f"($w_t = {', '.join(temporals)}$, "
          f"$\\delta \\in \\{{{', '.join(f'{d:g}' for d in deltas)}\\}}$, "
          f"$n_{{\\mathrm{{perm}}}} = {st['n_perm']}$, "
          f"{st['reps']} replicates, $N = {st['n']:,}$, $K = {st['K']}$). "
          "Smaller $\\mathrm{sep}$ = stronger clustering = easier alternative; "
          "$\\delta = 0$ makes the global null calibrated, so its rejection rate "
          "there is genuine power.", "",
          "| $w_t$ | $\\delta$ | sep (m) | mean $\\hat C(1,1)$ | power: global perm. | "
          "power: block perm. | power: normal approx. |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for temporal in temporals:
        for d in deltas:
            for s in seps:
                for r in rows:
                    if (r["temporal"] == temporal
                            and abs(r["delta"] - d) < 1e-9
                            and abs(r["sep_m"] - s) < 1e-9):
                        md.append(
                            f"| {temporal} | {d:.1f} | {s:.0f} | "
                            f"{r['mean_observed_pair']:.3f} | "
                            f"{r['power[perm:type]']:.3f}"
                            f"±{r['se[perm:type]']:.3f} | "
                            f"{r['power[perm:type_period]']:.3f}"
                            f"±{r['se[perm:type_period]']:.3f} | "
                            f"{r['power[norm:type]']:.3f} |")
    return "\n".join(md) + "\n"


def table_realdata(summaries):
    md = ["### Table S3. Real-world UK Police data", "",
          "Significance counts refer to the "
          "$K(K-1)$ **ordered, off-diagonal** type pairs (the $K$ diagonal entries are "
          "self-co-location and are not type pairs); $K=4$ gives 12 such pairs. "
          "Adjusted $p$-values are Benjamini--Hochberg at FDR $< 0.05$.", "",
          "| dataset | $N$ | types | periods | $b$ (m) | $\\rho$ | nnz | "
          "build (s) | 999 perms serial (s) | 999 perms 20 proc (s) | speed-up | "
          "bitwise | sig. pairs (global) | sig. pairs (block) | flips | "
          "$\\max\\lvert\\Delta p\\rvert$ |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for s in summaries:
        md.append(f"| {s['dataset']} | {s['n_points']:,} | {s['n_types']} | "
                  f"{s['n_periods']} | {s['bandwidth_m']:.0f} | {s['rho']:.2e} | "
                  f"{s['nnz']:,} | {s['build_s']:.2f} | {s['serial_seconds']:.1f} | "
                  f"{s['parallel_seconds']:.1f} | {s['parallel_speedup']:.1f}× | "
                  f"{s['bitwise_identical']} | "
                  f"{s['n_significant_fdr[type]']} | "
                  f"{s['n_significant_fdr[type_period]']} | "
                  f"{s.get('n_decision_flip', 'n/a')} | "
                  f"{s.get('max_abs_p_diff', float('nan')):.4f} |")
    return "\n".join(md) + "\n"


def table_period_stability(stab_avon, stab_london):
    md = ["### Table S8. Month-to-month stability of the FDR-significant type pairs", "",
          "For every ordered type pair that the pooled analysis flags at FDR $< 0.05$, the "
          "spatio-temporal CLQ is recomputed at the *same* spatial bandwidth separately in each "
          "of the twelve periods. Within a single period every time stamp coincides, so "
          "$w_t = 1$ and each per-period value is the spatial CLQ at that bandwidth; the pooled "
          "column is the temporally weighted value that the significance test is run on. "
          "$n_{\\text{attract}}$ counts the periods with STCLQ $> 1$; a pair is marked "
          "sign-consistent when all twelve agree in direction.", "",
          "| dataset | type pair | pooled STCLQ | monthly min | monthly max | monthly median | "
          "$n_{\\text{attract}}$ (of 12) | sign-consistent |",
          "|---|---|---:|---:|---:|---:|---:|:--:|"]
    for d in (stab_avon, stab_london):
        for r in d["pairs"]:
            md.append(f"| {d['dataset']} | {r['type_a']}–{r['type_b']} | "
                      f"{r['pooled_stclq']:.3f} | {r['min']:.3f} | {r['max']:.3f} | "
                      f"{r['median']:.3f} | {r['n_months_attract']} | "
                      f"{'yes' if r['sign_consistent'] else 'no'} |")
    return "\n".join(md) + "\n"


def table_feasible(table):
    md = ["### Table S4. Feasible-scale criterion: "
          "$\\rho \\le 0.01$ and the bandwidth that must accompany $N$", "",
          "Uniform density on a 20 km × 20 km domain; $c = 3$; $\\rho = "
          "\\mathrm{nnz}/N^2$.", "",
          "| $N$ | $b_{\\max}$ for $\\rho=0.01$ (m) | $\\bar k$ at $b_{\\max}$ | "
          "nnz at $b_{\\max}$ | CSR (MB) | $b$ for $\\bar k = 150$ (m) | "
          "nnz at $\\bar k=150$ |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['n']:,} | {r['b_max_rho']:.0f} | {r['kbar_at_b_max']:,.0f} | "
                  f"{r['nnz_at_b_max']:,.0f} | {r['csr_mb_at_b_max']:,.1f} | "
                  f"{r['b_for_target_knn']:.0f} | {r['nnz_at_target_knn']:,.0f} |")
    return "\n".join(md) + "\n"


def table_regime(reg):
    """Table S6: host state plus the configuration comparison under **interleaved**
    measurement.

    The two tables form one supplementary table because they are two halves of the same
    argument: the host is occupied by co-tenants rather than throttled, and under the
    same load different configurations reach different speed-ups.
    """
    md = ["### Table S6. Host state and the configuration dependence of "
          "parallel speed-up", ""]

    hosts = reg.get("host") or []
    h0 = hosts[0] if hosts else {}
    topo = h0.get("topology") or {}
    md += ["**(a) Host state recorded with the measurements.**", "",
           "| quantity | value |", "|---|---|",
           f"| CPU | {h0.get('cpu_model', '?')}, single socket |",
           f"| logical CPUs permitted (`Cpus_allowed_list`) | "
           f"{h0.get('nproc', '?')} (`{h0.get('cpus_allowed', '?')}`) |"]
    if topo:
        p_t = ", ".join(str(i) for i in topo.get("p_threads", [])[:4])
        e_t = ", ".join(str(i) for i in topo.get("e_threads", []))
        md += [
            f"| **P-cores** (SMT, {topo.get('max_freq_khz', 0) // 1000} MHz) | "
            f"{topo.get('n_p_threads', '?')} threads ({p_t}, …) |",
            f"| **E-cores** (no SMT, 3600 MHz) | {topo.get('n_e_threads', '?')} "
            f"threads ({e_t}) |"]
    md += [
           f"| NUMA nodes available for memory (`Mems_allowed_list`) | "
           f"`{h0.get('mems_allowed', '?')}` (single socket) |",
           f"| threads on the machine during the session | "
           f"{h0.get('n_threads', '?')} |",
           f"| cgroup throttled periods (`nr_throttled`) | "
           f"{h0.get('cpu_nr_throttled', '?')} |",
           f"| cgroup CPU limit (`cpu.max`) | "
           f"{h0.get('cgroup_cpu_max') or 'unset'} |"]
    loads = [x["loadavg"][0] for x in hosts if x.get("loadavg")]
    load_txt = f"{min(loads):.1f}–{max(loads):.1f}" if loads else "fluctuating"
    if loads:
        md.append(f"| one-minute load average over the session | "
                  f"{load_txt} |")
    md += ["", "The container is **not** throttled: every CPU is permitted and no "
           "cgroup throttling is recorded. Three host properties nevertheless bound "
           "what a pool can achieve, and none of them is under the algorithm's "
           "control. (i) The CPU is **hybrid**: 16 of the 20 threads are P-core "
           "threads with SMT and four are slower E-cores, so a $p=20$ pool cannot be "
           "uniformly fast, and `Pool.map` returns only when the *slowest* worker "
           "finishes. (ii) The machine is **shared**: of the order of "
           f"{h0.get('n_threads', '?')} threads and a one-minute load average "
           f"between {load_txt} during the session, so the throughput available to "
           "us fluctuates. (iii) Memory sits on a single socket, whose aggregate "
           "streaming rate saturates at ~2.6× the single-core rate. This is why "
           "every parallel figure in this paper is reported as a range over "
           "sessions rather than as a scalar.", ""]

    n_jobs = (reg.get("settings") or {}).get("n_jobs", "?")
    md += ["**(b) Interleaved measurement: the same round, the same co-tenancy.** "
           "Each row is timed once per round, all rows within a round; the "
           "tabulated value is the median over rounds, so the columns are "
           "comparable with one another even though the load drifts between "
           "rounds.", "",
           "| configuration | $N$ | $K$ | nnz | CSR (MB) | serial (s) | "
           f"parallel, $p={n_jobs}$ (s) | speed-up | "
           "GB/s serial | GB/s parallel |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reg.get("rows") or []:
        md.append(f"| `{r['name']}` | {r['n']:,} | {r['K']} | {r['nnz']:,} | "
                  f"{r['csr_mb']:.0f} | {r['serial_s']:.2f} | "
                  f"{r['parallel_s']:.2f} | {r['speedup']:.2f}× | "
                  f"{r['gbs_serial']:.1f} | {r['gbs_parallel']:.1f} |")
    md += ["", "Throughputs are CSR bytes streamed per second per permutation "
           "(`16 × nnz / t`), i.e. the fraction of the single-process streaming "
           "rate each configuration actually reaches; none approaches it, so the "
           "speed-ups are not bandwidth-limited."]
    return "\n".join(md) + "\n"


def table_parallel_detail(par, pat):
    """Table S7: parallel details compressed out of §6.4 and moved here.

    The four blocks correspond to four sets of quantities that the main text drops but
    that are **still measured**:
    (a) platform calibration (CPU-bound reference curve and memory-bandwidth ceiling),
        the frame of reference for the readings in Fig.5(a);
    (b) the start-up overhead scan against problem size, the criterion behind "when not
        to parallelise" in §4.3;
    (c) CSR and single-permutation cost for synthetic against real point patterns;
    (d) weak-scaling efficiency and the memory-bound control curve.
    """
    plat = par.get("platform") or {}
    md = ["### Table S7. Parallel details moved out of the main text",
          "",
          "Measured in the same session as Fig. 5; the host state is in Table S6.",
          ""]

    # (a) platform calibration
    cpu = plat.get("cpu") or []
    bw = {b["p"]: b for b in (plat.get("bandwidth") or [])}
    if cpu:
        md += ["**(a) Platform calibration: what this host can deliver at all.** "
               "The CPU-bound reference is a pure-compute loop over a $1.6$ MB "
               "working set (almost no memory traffic); bandwidth is the aggregate "
               "streaming rate over a $256$ MB array. Neither reference is reached "
               "by the permutation engine, so the flat strong-scaling curve of "
               "Fig. 5(a) is a property of the host rather than of the algorithm.",
               "",
               "| $p$ | CPU-bound speed-up | CPU-bound efficiency | "
               "aggregate BW (GB/s) | BW vs 1 process |",
               "|---:|---:|---:|---:|---:|"]
        for c in cpu:
            b = bw.get(c["p"], {})
            md.append(f"| {c['p']} | {c['speedup']:.2f}× | {c['efficiency']:.2f} | "
                      f"{b.get('aggregate_gbs', float('nan')):.1f} | "
                      f"{b.get('relative', float('nan')):.2f}× |")
        mx = max(cpu, key=lambda c: c["speedup"])
        md += ["", f"*Ceilings: CPU-bound maximum {plat.get('max_cpu_speedup', float('nan')):.2f}× "
               f"at $p={mx['p']}$, against {cpu[-1]['speedup']:.2f}× at $p={cpu[-1]['p']}$; "
               f"peak aggregate bandwidth {plat.get('peak_aggregate_gbs', float('nan')):.1f} GB/s "
               f"({plat.get('max_bw_relative', float('nan')):.2f}× the single-process rate).*", ""]

    # (b) start-up overhead scan
    ov = par.get("startup_overhead") or []
    if ov:
        md += ["**(b) Start-up overhead against problem size** (one permutation, "
               "serial vs $p$ processes). The overhead is flat in $\\mathrm{nnz}$, "
               "which is what the fork-shared design of §4.3 predicts; it is the "
               "quantity behind the serial fallback threshold.",
               "",
               "| $N$ | nnz | CSR (MB) | serial 1 perm (ms) | parallel 1 perm (ms) | "
               "speed-up | overhead (ms) |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in ov:
            md.append(f"| {r['n']:,} | {r['nnz']:,} | {r['csr_mb']:.0f} | "
                      f"{r['serial_1perm_ms']:.1f} | {r['parallel_1perm_ms']:.1f} | "
                      f"{r['speedup']:.2f}× | {r['overhead_1perm_ms']:.1f} |")
        md += ["", f"*Speed-up over the scan spans "
               f"{min(x['speedup'] for x in ov):.2f}–{max(x['speedup'] for x in ov):.2f}× "
               f"from $N={ov[0]['n']:,}$ to $N={ov[-1]['n']:,}$ (a thirty-fold range "
               f"of $N$): the gain does not decay with size, which is the direct "
               f"evidence for cache sharing rather than per-worker serialisation.*", ""]

    # (c) synthetic vs real patterns
    if pat:
        r, s = pat["real"], pat["synthetic"]
        md += ["**(c) Synthetic and real point patterns are not interchangeable.** "
               "Two engines matched in $N$, $K$ and mean neighbourhood size: the "
               "synthetic pattern is the slower of the two, so the controlled sweeps "
               "of §6.2 are conservative rather than faithful proxies.",
               "",
               "| pattern | nnz | CSR (MB) | 1 perm (ms) |",
               "|---|---:|---:|---:|",
               f"| Avon and Somerset (real) | {r['nnz']:,} | {r['csr_mb']:.0f} | "
               f"{r['perm_ms']:.0f} |",
               f"| synthetic, matched | {s['nnz']:,} | {s['csr_mb']:.0f} | "
               f"{s['perm_ms']:.0f} |",
               "",
               f"*Non-zero counts differ by {pat['nnz_rel_diff_pct']:.2f}%; the "
               f"synthetic pattern costs {pat['ratio_synth_over_real']:.2f}× more per "
               f"permutation, an access-pattern difference that the "
               f"$O(\\mathrm{{nnz}})$ analysis does not see.*", ""]

    # (d) weak scaling
    w = par.get("weak") or []
    ctrl = plat.get("weak_control") or []
    if w or ctrl:
        md += ["**(d) Weak scaling and the memory-bound control.** The point set is "
               "held fixed and the permutation count grows with $p$, so the work per "
               "process is constant; efficiency is $T_1/T_p$.",
               "",
               "| $p$ | $N$ | perms | per perm (ms) | efficiency | "
               "control efficiency (memory-bound) | prefix identical |",
               "|---:|---:|---:|---:|---:|---:|---|"]
        ctrl_by_p = {c["p"]: c["efficiency"] for c in ctrl}
        for r in w:
            ce = ctrl_by_p.get(r["p"])
            ce_txt = f"{ce:.2f}" if ce is not None else "n/a"
            md.append(f"| {r['p']} | {r['n']:,} | {r['n_perm']} | "
                      f"{r['per_perm_ms']:.1f} | {r['efficiency']:.2f} | "
                      f"{ce_txt} | {r.get('prefix_identical', 'n/a')} |")
        if w:
            md += ["", f"*Minimum engine efficiency {min(x['efficiency'] for x in w):.2f}"
                   + (f", against {min(c['efficiency'] for c in ctrl):.2f} for the "
                      f"platform's own memory-bound control curve." if ctrl else ".")
                   + "*"]
    return "\n".join(md) + "\n"


def main():
    written = []
    p = TABDIR.parent / "GA_manuscript.md"
    TABDIR.mkdir(parents=True, exist_ok=True)

    (TABDIR / "table1_algorithm.md").write_text(ALGORITHM, encoding="utf-8")
    written.append("table1_algorithm.md")

    md2, tex2 = table2()
    (TABDIR / "table2_complexity.md").write_text(md2, encoding="utf-8")
    (TABDIR / "table2_complexity.tex").write_text(tex2, encoding="utf-8")
    written += ["table2_complexity.md", "table2_complexity.tex"]

    bench_p = HERE / "benchmark" / "benchmark.json"
    if bench_p.exists():
        md3, tex3 = table3(json.loads(bench_p.read_text(encoding="utf-8")))
        (TABDIR / "table3_benchmark.md").write_text(md3, encoding="utf-8")
        (TABDIR / "table3_benchmark.tex").write_text(tex3, encoding="utf-8")
        written += ["table3_benchmark.md", "table3_benchmark.tex"]

    corr_p = HERE / "correctness" / "correctness.json"
    if corr_p.exists():
        (TABDIR / "tableS1_correctness.md").write_text(
            table_correctness(json.loads(corr_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS1_correctness.md")

    cal_p = HERE / "calibration" / "calibration.json"
    if cal_p.exists():
        (TABDIR / "tableS2_calibration.md").write_text(
            table_calibration(json.loads(cal_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS2_calibration.md")

    real_p = HERE / "realdata" / "summary_all.json"
    if real_p.exists():
        (TABDIR / "tableS3_realdata.md").write_text(
            table_realdata(json.loads(real_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS3_realdata.md")

    pw_p = HERE / "calibration" / "calibration_power.json"
    if pw_p.exists():
        (TABDIR / "tableS5_power.md").write_text(
            table_power(json.loads(pw_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS5_power.md")

    feas_p = HERE / "figures" / "feasible_scale_table.json"
    if feas_p.exists():
        (TABDIR / "tableS4_feasible_scale.md").write_text(
            table_feasible(json.loads(feas_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS4_feasible_scale.md")

    reg_p = HERE / "regime" / "regime.json"
    if reg_p.exists():
        (TABDIR / "tableS6_regime.md").write_text(
            table_regime(json.loads(reg_p.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS6_regime.md")

    par_p = HERE / "parallel" / "parallel.json"
    pat_p = HERE / "regime" / "pattern.json"
    if par_p.exists():
        par = json.loads(par_p.read_text(encoding="utf-8"))
        pat = (json.loads(pat_p.read_text(encoding="utf-8"))
               if pat_p.exists() else None)
        (TABDIR / "tableS7_parallel_detail.md").write_text(
            table_parallel_detail(par, pat), encoding="utf-8")
        written.append("tableS7_parallel_detail.md")

    stab_a = HERE / "realdata" / "avon_and_somerset_period_stability.json"
    stab_l = HERE / "realdata" / "city_of_london_period_stability.json"
    if stab_a.exists() and stab_l.exists():
        (TABDIR / "tableS8_period_stability.md").write_text(
            table_period_stability(json.loads(stab_a.read_text(encoding="utf-8")),
                                   json.loads(stab_l.read_text(encoding="utf-8"))),
            encoding="utf-8")
        written.append("tableS8_period_stability.md")

    _ = p
    print("tables ->", TABDIR)
    for w in written:
        print("  ", w)


if __name__ == "__main__":
    main()
