"""
Debug harness for cqrlib.tri_barrier — single-threaded + memory tracking.

Why num_threads=1:
  mp_pandas_obj dispatches to process_jobs_ (sequential, main process only)
  when num_threads == 1, so breakpoints work and memory you observe belongs
  to the process you are debugging (AFML snippet 20.8).

Memory status is reported via tracemalloc (stdlib): peak traced memory plus
the top allocation sites during the tri_barrier call.

Usage:
  python scripts/debug_tri_barrier.py   # run from the repo root

With the VS Code debugger:
  1. Open this file, set a breakpoint e.g. inside
     cqrlib/cqrlib/Labels/triple_barrier_method.py -> _pt_sl_t1 (the loop body).
  2. Run and Debug (F5). Inspect locals / memory in the debugger.
"""

import time
import tracemalloc

import pandas as pd
import cqrlib as rs

DATA = "./sample-data/dollar_bars.csv"


def main():
    print("loading data...")
    dollar = pd.read_csv(DATA, sep=",", header=0, parse_dates=True, index_col=["date_time"])
    # pandas 3.0 parses datetimes as datetime64[us]; cqrlib requires datetime64[ns]
    dollar.index = dollar.index.as_unit("ns")

    d_vol = rs.vol(dollar["close"], span0=50)
    events = rs.cs_filter(dollar["close"], limit=d_vol.mean())
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
    tracemalloc.stop()

    print(f"\ntri_barrier done in {elapsed:.2f}s")
    print(f"tb shape: {tb.shape}")
    print(f"\n--- memory during tri_barrier ---")
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

    return tb


if __name__ == "__main__":
    tb = main()
