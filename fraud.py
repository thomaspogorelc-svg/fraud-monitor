"""Rule-based fraud risk scoring engine (no Streamlit dependency, easy to test)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["txn_id", "customer_id", "timestamp", "amount", "merchant_category", "country"]
HIGH_RISK_CATEGORIES = {"crypto", "gambling", "wire_transfer", "gift_cards"}


class DataError(ValueError):
    """Raised when the transaction data cannot be used."""


@dataclass
class RuleConfig:
    # Rule parameters
    amount_z_threshold: float = 3.5      # robust z-score at which the amount rule fully triggers
    velocity_window_min: int = 60        # window for counting transactions
    velocity_count: int = 5              # transactions in window that fully trigger the rule
    night_start: int = 0                 # unusual-hour window, inclusive start hour
    night_end: int = 5                   # inclusive end hour
    large_multiple: float = 3.0          # "large" = this many times the customer's median amount
    # Rule weights (max points each rule can add). They sum to more than 100 on purpose:
    # several rules usually have to fire together to reach a high score, and the total is capped at 100.
    w_amount: float = 45.0
    w_velocity: float = 40.0
    w_geo: float = 25.0
    w_night: float = 15.0
    w_category: float = 25.0

    def as_dict(self) -> dict:
        return asdict(self)


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Check schema, coerce types, drop unusable rows. Returns (clean_df, warnings)."""
    if df is None or df.empty:
        raise DataError("The dataset is empty.")
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required column(s): {', '.join(missing)}. "
                        f"Required: {', '.join(REQUIRED_COLUMNS)}.")
    warnings: list[str] = []
    n0 = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["amount"] > 0]
    dropped = n0 - len(df)
    if dropped:
        warnings.append(f"Dropped {dropped} row(s) with missing/invalid timestamp, amount, or fields.")
    if len(df) < 20:
        raise DataError("Too few valid transactions (need at least 20) after cleaning.")
    df["customer_id"] = df["customer_id"].astype(str)
    df["merchant_category"] = df["merchant_category"].astype(str).str.lower()
    df["country"] = df["country"].astype(str).str.upper()
    return df.reset_index(drop=True), warnings


def add_features(df: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Add per-customer baselines and velocity features."""
    out = df.copy()
    g = out.groupby("customer_id")["amount"]
    out["cust_median"] = g.transform("median")
    med = out["cust_median"]
    abs_dev = (out["amount"] - med).abs()
    mad = abs_dev.groupby(out["customer_id"]).transform("median")
    # floor the MAD so customers with very uniform spending don't produce absurd z-scores
    scale = np.maximum(mad, np.maximum(0.1 * med, 1.0))
    out["amount_z"] = 0.6745 * (out["amount"] - med) / scale
    out["amount_ratio"] = out["amount"] / med

    home = out.groupby("customer_id")["country"].agg(lambda s: s.mode().iloc[0])
    out["home_country"] = out["customer_id"].map(home)

    out["hour"] = out["timestamp"].dt.hour

    s = out.sort_values(["customer_id", "timestamp"])
    cnt = (s.set_index("timestamp").groupby("customer_id")["amount"]
             .rolling(f"{int(cfg.velocity_window_min)}min").count())
    s["txns_in_window"] = cnt.values.astype(int)
    out["txns_in_window"] = s["txns_in_window"].reindex(out.index)
    return out


def score_transactions(df: pd.DataFrame, cfg: RuleConfig | None = None) -> pd.DataFrame:
    """Return df with rule intensities, a 0-100 risk_score and a text explanation."""
    cfg = cfg or RuleConfig()
    f = add_features(df, cfg)

    z = f["amount_z"].to_numpy()
    amount_i = np.where(z >= cfg.amount_z_threshold, 1.0, np.where(z >= cfg.amount_z_threshold / 2, 0.5, 0.0))

    n = f["txns_in_window"].to_numpy()
    vel_i = np.where(n >= cfg.velocity_count, 1.0, np.where(n == cfg.velocity_count - 1, 0.5, 0.0))
    if cfg.velocity_count <= 1:
        vel_i = np.where(n >= cfg.velocity_count, 1.0, 0.0)

    geo_i = (f["country"] != f["home_country"]).astype(float).to_numpy()

    h = f["hour"]
    if cfg.night_start <= cfg.night_end:
        is_night = (h >= cfg.night_start) & (h <= cfg.night_end)
    else:  # window wraps midnight, e.g. 22 -> 4
        is_night = (h >= cfg.night_start) | (h <= cfg.night_end)
    night_i = is_night.astype(float).to_numpy()

    risky = f["merchant_category"].isin(HIGH_RISK_CATEGORIES)
    large = f["amount_ratio"] >= cfg.large_multiple
    cat_i = np.where(risky & large, 1.0, np.where(risky, 0.4, 0.0))

    points = {
        "pts_amount": cfg.w_amount * amount_i,
        "pts_velocity": cfg.w_velocity * vel_i,
        "pts_geo": cfg.w_geo * geo_i,
        "pts_night": cfg.w_night * night_i,
        "pts_category": cfg.w_category * cat_i,
    }
    for k, v in points.items():
        f[k] = np.round(v, 1)
    f["risk_score"] = np.clip(sum(points.values()), 0, 100).round(1)

    reasons = pd.DataFrame({
        "a": np.where(amount_i > 0,
                      "Amount $" + f["amount"].round(2).astype(str) + " is " +
                      f["amount_ratio"].round(1).astype(str) + "x this customer's median ($" +
                      f["cust_median"].round(2).astype(str) + ") [robust z=" + f["amount_z"].round(1).astype(str) + "]", ""),
        "b": np.where(vel_i > 0,
                      f["txns_in_window"].astype(str) + " transactions by this customer within " +
                      str(int(cfg.velocity_window_min)) + " min", ""),
        "c": np.where(geo_i > 0, "Country " + f["country"] + " differs from home country " + f["home_country"], ""),
        "d": np.where(night_i > 0, "Occurred at unusual hour (" + f["hour"].astype(str) + ":00)", ""),
        "e": np.where(cat_i > 0, "High-risk category '" + f["merchant_category"] + "'" +
                      np.where(large, " with a large amount", ""), ""),
    }, index=f.index)
    f["explanation"] = reasons.apply(lambda r: "; ".join(x for x in r if x), axis=1)
    f.loc[f["explanation"] == "", "explanation"] = "No rules triggered"
    return f


def flag(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Transactions with risk_score >= threshold, highest risk first."""
    return scored[scored["risk_score"] >= threshold].sort_values("risk_score", ascending=False)


def evaluate(scored: pd.DataFrame, threshold: float, label_col: str = "synthetic_fraud_label") -> dict | None:
    """Precision/recall vs. the synthetic ground-truth label, if present."""
    if label_col not in scored.columns:
        return None
    y = pd.to_numeric(scored[label_col], errors="coerce").fillna(0).astype(int)
    pred = (scored["risk_score"] >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
