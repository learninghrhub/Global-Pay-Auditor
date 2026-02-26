from __future__ import annotations

import pandas as pd


def build_peer_group(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Build a peer group label from a list of grouping columns."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(["ALL"] * len(df), index=df.index)
    return df[cols].astype(str).agg(" | ".join, axis=1)


def assign_peer_groups(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Assign progressively broader peer groups until minimum size is met."""
    out = df.copy()

    out["peer_group"] = build_peer_group(out, ["grade", "job_family", "country"])
    counts = out["peer_group"].value_counts()
    small = out["peer_group"].map(counts) < min_n

    if small.any():
        out.loc[small, "peer_group"] = build_peer_group(out.loc[small], ["grade", "job_family"])
        counts = out["peer_group"].value_counts()
        small = out["peer_group"].map(counts) < min_n

    if small.any():
        out.loc[small, "peer_group"] = build_peer_group(out.loc[small], ["grade", "country"])
        counts = out["peer_group"].value_counts()
        small = out["peer_group"].map(counts) < min_n

    if small.any():
        out.loc[small, "peer_group"] = build_peer_group(out.loc[small], ["grade"])

    return out
