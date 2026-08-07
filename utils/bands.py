"""Bollinger-band helpers shared across the AFML notebooks."""

import pandas as pd


def bband_frac(data: pd.DataFrame, window: int = 21, width: float = 0.001):
    """Bands as a fixed fraction of the EMA — price-level form, not a moment."""
    ewm = data["close"].ewm(span=window)  # single ewm object reused for avg & std
    avg = ewm.mean()
    std = avg * width
    upper, lower = avg + std, avg - std
    return avg, upper, lower, std


def bband_std(data: pd.DataFrame, window: int = 21, width: float = 2.0):
    """Bands as a multiple of the EMA's std (sqrt of the 2nd central moment)."""
    ewm = data["close"].ewm(span=window)  # single ewm object reused for avg & std
    avg = ewm.mean()
    std = ewm.std()
    upper, lower = avg + width * std, avg - width * std
    return avg, upper, lower, std


def side_pick(data: pd.DataFrame):
    """Label each bar by the band it touches: -1 upper (short), +1 lower (long)."""
    for i, idx in enumerate(data.index):
        if data["close"].iloc[i] >= data["upper"].iloc[i]:
            data.loc[idx, "side"] = -1
        elif data["close"].iloc[i] <= data["lower"].iloc[i]:
            data.loc[idx, "side"] = 1
    return data
