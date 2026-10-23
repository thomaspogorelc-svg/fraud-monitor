import pandas as pd
import pytest

from data_gen import generate_synthetic_transactions
from fraud import DataError, RuleConfig, evaluate, flag, score_transactions, validate_data


def _base(n=40):
    ts = pd.date_range("2026-07-01 12:00", periods=n, freq="D")
    return pd.DataFrame({
        "txn_id": [f"T{i}" for i in range(n)], "customer_id": "C1", "timestamp": ts,
        "amount": 20.0 + (pd.Series(range(n)) % 5), "merchant_category": "groceries", "country": "US",
    })


def test_large_amount_flagged():
    df = _base()
    df.loc[10, "amount"] = 900.0
    s = score_transactions(df)
    assert s.loc[10, "risk_score"] >= 45
    assert "median" in s.loc[10, "explanation"]
    assert s.loc[0, "risk_score"] == 0


def test_velocity_rule():
    df = _base()
    burst = pd.DataFrame({
        "txn_id": [f"B{i}" for i in range(6)], "customer_id": "C1",
        "timestamp": pd.Timestamp("2026-09-01 14:00") + pd.to_timedelta(range(6), unit="min"),
        "amount": 21.0, "merchant_category": "retail", "country": "US"})
    s = score_transactions(pd.concat([df, burst], ignore_index=True))
    assert s["txns_in_window"].max() == 6
    assert (s["pts_velocity"] > 0).sum() >= 2


def test_foreign_night_risky():
    df = _base()
    df.loc[5, ["country", "merchant_category"]] = ["GB", "crypto"]
    df.loc[5, "timestamp"] = pd.Timestamp("2026-07-06 03:00")
    s = score_transactions(df)
    assert s.loc[5, "pts_geo"] == 25 and s.loc[5, "pts_night"] == 15 and s.loc[5, "pts_category"] > 0


def test_validation_errors():
    with pytest.raises(DataError):
        validate_data(pd.DataFrame())
    with pytest.raises(DataError):
        validate_data(pd.DataFrame({"a": [1]}))
    bad = _base()
    bad["amount"] = bad["amount"].astype(object)
    bad["timestamp"] = bad["timestamp"].astype(object)
    bad.loc[0, "amount"] = "oops"
    bad.loc[1, "timestamp"] = "not a date"
    clean, warns = validate_data(bad)
    assert len(clean) == 38 and warns


def test_synthetic_end_to_end():
    d = generate_synthetic_transactions(100, 60, seed=1)
    clean, _ = validate_data(d)
    s = score_transactions(clean)
    assert s["risk_score"].between(0, 100).all()
    ev = evaluate(s, 50)
    assert ev["recall"] > 0.4 and ev["precision"] > 0.7
    assert len(flag(s, 50)) > 0
