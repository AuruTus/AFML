"""
Debug harness for cqrlib.tri_barrier + meta_label — single-threaded + memory tracking.

Why num_threads=1:
  mp_pandas_obj dispatches to process_jobs_ (sequential, main process only)
  when num_threads == 1, so breakpoints work and memory you observe belongs
  to the process you are debugging (AFML snippet 20.8).

Memory status is reported via tracemalloc (stdlib): peak traced memory plus
the top allocation sites during the tri_barrier / meta_label calls.

Usage:
  python scripts/debug_tri_barrier.py   # run from the repo root

With the VS Code debugger:
  1. Open this file, set a breakpoint e.g. inside
     cqrlib/cqrlib/Labels/triple_barrier_method.py:
       - _pt_sl_t1 (the loop body)      -> triple barrier formation
       - meta_label (the 'ret'/'bin' lines) -> labeling / drop path
  2. Run and Debug (F5). Inspect locals / memory in the debugger.
"""

import time
import tracemalloc

import pandas as pd
import cqrlib as rs

DATA = "./sample-data/dollar_bars.csv"


def _report_memory(snap0, snap1, peak, elapsed, label):
    """Print a tracemalloc comparison between snap0 and snap1 for a call."""
    print(f"\n{label} done in {elapsed:.2f}s")
    print(f"tracemalloc peak: {peak / 1e6:.1f} MB")
    print("top allocation sites:")
    for stat in snap1.compare_to(snap0, "lineno")[:10]:
        print(stat)
    try:
        import psutil

        proc = psutil.Process()
        print(f"process RSS now: {proc.memory_info().rss / 1e6:.1f} MB")
    except ImportError:
        print("(psutil not installed - skipping RSS check)")


def main():
    print("loading data...")
    dollar = pd.read_csv(DATA, sep=",", header=0, parse_dates=True, index_col=["date_time"])
    # pandas 3.0 parses datetimes as datetime64[us]; cqrlib requires datetime64[ns]
    dollar.index = dollar.index.as_unit("ns")

    d_vol = rs.vol(dollar["close"], span0=50)
    # d_vol is a return; cs_filter diffs are price points, so scale by price level
    events = rs.cs_filter(dollar["close"], limit=d_vol.mean() * dollar["close"].mean())
    vb = rs.vert_barrier(data=dollar["close"], events=events, period="days", freq=1)
    print(f"data rows={dollar.shape[0]}  events={events.shape[0]}  vb={vb.shape[0]}")

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    t0 = time.time()

    # num_threads=1 => single process, debugger-friendly (change to 3 to mirror the notebook)
    tb = rs.tri_barrier(
        data=dollar["close"],
        events=events,
        trgt=d_vol,
        min_req=0.002,
        num_threads=1,
        ptSl=[1, 1],
        t1=vb,
        side=None,
    )

    elapsed = time.time() - t0
    snap1 = tracemalloc.take_snapshot()
    peak = tracemalloc.get_traced_memory()[1]
    _report_memory(snap0, snap1, peak, elapsed, "tri_barrier")
    print(f"tb shape: {tb.shape}")

    # --- meta_label (mirrors notebook cell: AFML 3.1) ---
    snap0 = tracemalloc.take_snapshot()
    t0 = time.time()

    # Breakpoint target: cqrlib/Labels/triple_barrier_method.py -> meta_label,
    # e.g. the 'out["ret"] = ...' / "if 'side' in events_" lines.
    m_label = rs.meta_label(data=dollar["close"], events=tb, drop=False)

    elapsed = time.time() - t0
    snap1 = tracemalloc.take_snapshot()
    peak = tracemalloc.get_traced_memory()[1]
    _report_memory(snap0, snap1, peak, elapsed, "meta_label")
    print(f"m_label shape: {m_label.shape}")
    print("\n'bin' value counts:")
    print(m_label["bin"].value_counts(normalize=True))

    # --- drop_label (drop rare labels, min_pct=0.05 as in notebook) ---
    drop_meta_label = rs.drop_label(events=m_label, min_pct=0.05)
    print(f"\ndrop_label: kept {drop_meta_label.shape[0]} of {m_label.shape[0]} rows")
    print(drop_meta_label["bin"].value_counts())

    tracemalloc.stop()

    return tb


if __name__ == "__main__":
    tb = main()
