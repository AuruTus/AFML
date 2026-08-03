"""
Compare the CUSUM event filter on absolute prices vs log prices (AFML 3.1).

Why: cs_filter's docstring forbids replacing data.diff() with
np.log(data).diff(). This script visualizes the difference:

  - the two diff series live in different units (points vs returns)
  - event frequency and trigger-move sizes differ between transforms
  - the trigger-move-over-time scatter checks the homoscedasticity claim

Usage:
  python scripts/compare_log_vs_abs.py   # run from the repo root

Output: figure saved to scripts/output/log_vs_abs.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cqrlib as rs

DATA = "./sample-data/dollar_bars.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def _cusum(prices, lim):
    """Core CUSUM loop (same logic as cs_filter).

    Returns: (events, trig) where events is the DatetimeIndex of trigger
    timestamps and trig is the Series of cumulative move sizes that crossed
    the threshold (in the same units as `prices`).
    """
    diff = prices.diff()
    idx, trig, _up, _dn = [], [], 0.0, 0.0
    for i in diff.index[1:]:
        _up = max(0.0, float(_up + diff.loc[i]))
        _dn = min(0.0, float(_dn + diff.loc[i]))
        if _up >= lim:
            trig.append(_up)
            idx.append(i)
            _up = 0.0
        elif _dn <= -lim:
            trig.append(-_dn)
            idx.append(i)
            _dn = 0.0
    return pd.DatetimeIndex(idx), pd.Series(trig, index=idx)


def cusum_abs(data, limit):
    """CUSUM filter on absolute prices. Returns (events, trig in price points).

    Units: `limit` must be in PRICE units (points), e.g. vol(return) * price level.
    Passing a return-units threshold like d_vol.mean() (~0.0055) here is a
    units mismatch: relative to a ~2000-point price it is ~0.0003%, so nearly
    every bar triggers an event (22,890 out of 24,079 in this dataset).
    """
    return _cusum(data, float(limit))


def cusum_log(data, limit):
    """CUSUM filter on log prices. Returns (events, trig in log-return units).

    Units: `limit` must be in LOG-RETURN units, e.g. np.log(1 + d_vol.mean()).
    This is the scale-free counterpart of cusum_abs with a price-scaled limit;
    with equivalent thresholds both produce nearly identical event sets.
    """
    return _cusum(np.log(data), float(limit))


def main():
    print("loading data...")
    dollar = pd.read_csv(DATA, sep=",", header=0, parse_dates=True, index_col=["date_time"])
    # pandas 3.0 parses datetimes as datetime64[us]; cqrlib requires datetime64[ns]
    dollar.index = dollar.index.as_unit("ns")

    d_vol = rs.vol(dollar["close"], span0=50)
    # NOTE on units: d_vol is a RETURN (0.0055), but data.diff() on absolute
    # prices is in POINTS. The same 'typical daily move' must be expressed in
    # each transform's units for a fair comparison:
    limit_ret = d_vol.mean()  # return units (0.0055)
    mean_px = dollar["close"].mean()
    limit_px = limit_ret * mean_px  # same move in price points (11.38)

    # variant 1: abs prices, threshold in points (the 'fair' abs usage)
    ev_abs, trig_abs = cusum_abs(dollar["close"], limit_px)
    # variant 2: abs prices, threshold in return units (notebook as-is -> floods)
    ev_abs_asis, trig_abs_asis = cusum_abs(dollar["close"], limit_ret)
    # variant 3: log prices, threshold in log-return units (fair log usage)
    # ln(1 + r) converts the simple-return move to log-return units (≈ r for small r)
    ev_log, trig_log = cusum_log(dollar["close"], np.log(1 + limit_ret))

    # trigger moves in % units so the variants are comparable
    pct_abs = trig_abs / dollar["close"].loc[ev_abs] * 100  # points -> %
    pct_abs_asis = trig_abs_asis / dollar["close"].loc[ev_abs_asis] * 100
    pct_log = np.expm1(trig_log) * 100  # log return -> %

    print(f"limit: {limit_ret:.4f} (returns) == {limit_px:.2f} (price points)")
    print(f"abs events, fair limit in points   : {len(ev_abs):6d}  (trigger %: {pct_abs.mean():.3f} ± {pct_abs.std():.3f})")
    print(
        f"abs events, notebook limit (returns): {len(ev_abs_asis):6d}  (trigger %: {pct_abs_asis.mean():.3f} ± {pct_abs_asis.std():.3f})"
    )
    print(f"log events, fair limit in log-ret  : {len(ev_log):6d}  (trigger %: {pct_log.mean():.3f} ± {pct_log.std():.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(dollar["close"].diff().dropna(), bins=100, density=True, alpha=0.7)
    axes[0, 0].set_title("distribution: absolute price diff (points)")

    axes[0, 1].hist(np.log(dollar["close"]).diff().dropna(), bins=100, density=True, alpha=0.7)
    axes[0, 1].set_title("distribution: log price diff (returns)")

    axes[1, 0].hist(pct_abs, bins=50, density=True, alpha=0.6, label=f"abs (n={len(ev_abs)})")
    axes[1, 0].hist(pct_log, bins=50, density=True, alpha=0.6, label=f"log (n={len(ev_log)})")
    axes[1, 0].set_title("trigger move size at events (%) — fair thresholds")
    axes[1, 0].legend()

    axes[1, 1].scatter(ev_abs, pct_abs, s=4, alpha=0.4, label="abs", color="C0")
    axes[1, 1].scatter(ev_log, pct_log, s=4, alpha=0.4, label="log", color="C1")
    axes[1, 1].set_title("trigger move size over time (%) — homoscedasticity check")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUT, "log_vs_abs.png")
    plt.savefig(out_path, dpi=120)
    print("figure saved:", out_path)


if __name__ == "__main__":
    main()
