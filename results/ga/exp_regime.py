"""State dependence of the parallel speed-up: interleaved, multi-round medians on a
single host.

Motivation
----------
The control sweep of §6.4 ($N=10^5$, $K=8$) and the real data of §6.6 (Avon:
$N=1.39\\times10^5$, $K=4$) give about $1.8\\times$ and $4.7\\times$ respectively on the
**same machine**. The two differ by nearly a factor of three, which cannot be
attributed to the algorithm alone, nor to the load alone: the gap has to be shown to
come from the **configuration** first.

Approach
--------
The key is **interleaving**: measuring the serial and the parallel run of one
configuration to completion before moving to the next leaves the two configurations
exposed to different host load, and the measured difference may be load drift. Each
"round" of this script measures the serial and the parallel run of **every
configuration** in a fixed order, so within a round all configurations share the same
stretch of load; the median over several rounds then separates the "configuration
effect" from the "time-varying load of the shared host".

    PYTHONPATH=src python results/ga/exp_regime.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

import exp_parallel as ep          # reuse build_engine / time_perm

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "regime"


def _read(p, default=None):
    try:
        return pathlib.Path(p).read_text(encoding="utf-8").strip()
    except Exception:
        return default


def cpu_topology():
    """Identify a hybrid (P+E core) topology.

    12th-generation Core and later desktop and mobile CPUs are **hybrid**: a few
    high-frequency performance cores (P-cores, with SMT) plus a few low-frequency
    efficiency cores (E-cores, without SMT). The total thread count reported as the
    "logical core count" therefore **overstates** the homogeneous compute available:
    placing a worker on an E-core slows that worker down noticeably, and the
    completion time of ``Pool.map`` is set by the **slowest** worker. Splitting the
    threads into two classes by their maximum frequency identifies the topology.
    """
    tc = pathlib.Path("/sys/devices/system/cpu")
    freq, p_cores, e_cores = {}, [], []
    for d in sorted(tc.glob("cpu[0-9]*"),
                    key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 0):
        try:
            i = int(d.name[3:])
        except ValueError:
            continue
        try:
            f = int((d / "cpufreq" / "cpuinfo_max_freq").read_text().strip())
        except Exception:
            continue
        freq[i] = f
    if not freq:
        return {}
    top = max(freq.values())
    for i, f in freq.items():
        (p_cores if f >= top * 0.9 else e_cores).append(i)
    return {"max_freq_khz": top, "freq_khz": freq,
            "p_threads": p_cores, "e_threads": e_cores,
            "n_p_threads": len(p_cores), "n_e_threads": len(e_cores)}


def host_state():
    """Record the host state: on a shared machine the comparability of parallel
    results depends on these quantities.

    ``nr_throttled``/``cgroup cpu.max`` serve to show that "slow" means **co-tenant
    contention** rather than this container being throttled; ``Mems_allowed_list``
    shows that the usable memory controllers cover a single NUMA node, which bears
    directly on the aggregate bandwidth ceiling measured on other machines.
    """
    st = {"nproc": os.cpu_count()}
    status = _read("/proc/self/status", "")
    for line in status.splitlines():
        if line.startswith("Cpus_allowed_list"):
            st["cpus_allowed"] = line.split(":", 1)[1].strip()
        elif line.startswith("Mems_allowed_list"):
            st["mems_allowed"] = line.split(":", 1)[1].strip()
    la = _read("/proc/loadavg", "")
    if la:
        parts = la.split()
        st["loadavg"] = [float(parts[0]), float(parts[1]), float(parts[2])]
        # the 4th field looks like "5/3292": runnable threads / total threads; the
        # total is kept
        frac = parts[3] if len(parts) > 3 else ""
        st["n_threads_raw"] = frac
        try:
            st["n_runnable"], st["n_threads"] = (int(x) for x in frac.split("/"))
        except Exception:
            pass
    cs = _read("/sys/fs/cgroup/cpu.stat", "")
    if cs:
        for line in cs.splitlines():
            k, _, v = line.partition(" ")
            if k in ("nr_throttled", "throttled_usec",
                     "core_sched.force_idle_usec"):
                st[f"cpu_{k}"] = v.strip()
    st["cgroup_cpu_max"] = _read("/sys/fs/cgroup/cpu.max")
    st["cpu_model"] = None
    for line in _read("/proc/cpuinfo", "").splitlines():
        if line.lower().startswith("model name"):
            st["cpu_model"] = line.split(":", 1)[1].strip()
            break
    st["topology"] = cpu_topology()
    return st

#: (label, build_engine arguments). The first two correspond to §6.4, the third to the
#: Avon data of §6.6, the fourth reduces K to 4 while keeping N, and the fifth is the
#: City small-working-set control of §6.6.
CONFIGS = [
    ("sweep_K8_N100k", dict(n=100_000, K=8, knn=150)),
    ("K8_N139k",       dict(n=139_001, K=8, knn=150)),
    ("avon_K4_N139k",  dict(n=139_001, K=4, knn=137)),
    ("K4_N100k",       dict(n=100_000, K=4, knn=150)),
    ("city_K4_N16k",   dict(n=15_776,  K=4, knn=184)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"Interleaved measurement: n_perm={args.n_perm}  n_jobs={args.n_jobs}  "
          f"rounds={args.rounds}\n")

    engines = []
    for name, cfg in CONFIGS:
        eng, bw = ep.build_engine(cfg["n"], target_knn=cfg["knn"],
                                  K=cfg["K"], seed=0)
        sim_bytes = 16 * eng.W.nnz          # CSR bytes every permutation must stream
        engines.append((name, cfg, eng, bw, sim_bytes))
        print(f"{name:>16}  N={cfg['n']:>7,}  K={cfg['K']}  "
              f"nnz={eng.W.nnz:>11,}  rho={eng.W.nnz / cfg['n'] ** 2:.3e}  "
              f"CSR={sim_bytes / 2 ** 20:7.1f} MB  b={bw:.1f} m")

    plan = np.arange(10_000, 10_000 + args.n_perm, dtype=np.int64)
    rec = {name: {"serial": [], "parallel": [], "gbs_serial": [],
                  "gbs_parallel": []} for name, *_ in engines}

    host0 = host_state()
    print(f"\nHost state: {host0.get('cpu_model')}")
    print(f"  nproc={host0['nproc']}  cpus_allowed={host0.get('cpus_allowed')}  "
          f"mems_allowed={host0.get('mems_allowed')}  "
          f"loadavg={host0.get('loadavg')}  threads={host0.get('n_threads')}")
    print(f"  cpu_throttled={host0.get('cpu_nr_throttled')}  "
          f"cpu.max={host0.get('cgroup_cpu_max')}")

    hosts = [host0]
    for r in range(args.rounds):
        print(f"\n--- round {r + 1}/{args.rounds} ---")
        for name, cfg, eng, bw, sb in engines:
            ts, _ = ep.time_perm(eng, plan, 1, repeat=1)
            tp, _ = ep.time_perm(eng, plan, args.n_jobs, repeat=1)
            rec[name]["serial"].append(ts)
            rec[name]["parallel"].append(tp)
            rec[name]["gbs_serial"].append(sb / ts / 1e9 * args.n_perm)
            rec[name]["gbs_parallel"].append(sb / tp / 1e9 * args.n_perm)
            print(f"  {name:>16}  serial {ts:7.2f}s  par(p={args.n_jobs}) "
                  f"{tp:7.2f}s  spdup {ts / tp:5.2f}x  "
                  f"({sb / ts * args.n_perm / 1e9:5.2f} / "
                  f"{sb / tp * args.n_perm / 1e9:5.2f} GB/s)")
        h = host_state()
        hosts.append(h)
        print(f"  [host after round {r + 1}] loadavg={h.get('loadavg')}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{'config':>16} {'N':>9} {'K':>2} {'nnz':>11} {'CSR MB':>8} "
          f"{'serial s':>9} {'par s':>8} {'speed-up':>9} {'GB/s ser':>9} "
          f"{'GB/s par':>9}")
    for name, cfg, eng, bw, sb in engines:
        d = rec[name]
        ts = float(np.median(d["serial"]))
        tp = float(np.median(d["parallel"]))
        gs = float(np.median(d["gbs_serial"]))
        gp = float(np.median(d["gbs_parallel"]))
        print(f"{name:>16} {cfg['n']:>9,} {cfg['K']:>2} {eng.W.nnz:>11,} "
              f"{sb / 2 ** 20:>8.1f} {ts:>9.2f} {tp:>8.2f} {ts / tp:>8.2f}x "
              f"{gs:>9.2f} {gp:>9.2f}")
        rows.append({"name": name, "n": cfg["n"], "K": cfg["K"],
                     "nnz": int(eng.W.nnz), "csr_mb": sb / 2 ** 20,
                     "serial_s": ts, "parallel_s": tp, "speedup": ts / tp,
                     "gbs_serial": gs, "gbs_parallel": gp,
                     "raw_serial": d["serial"], "raw_parallel": d["parallel"]})

    (OUT / "regime.json").write_text(
        json.dumps({"settings": vars(args), "host": hosts,
                    "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nresults -> {OUT / 'regime.json'}")


if __name__ == "__main__":
    main()
