"""Synthetic transaction generator. No real customer data is used anywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["groceries", "restaurants", "fuel", "retail", "travel", "utilities", "entertainment", "healthcare"]
CAT_P = [0.25, 0.18, 0.12, 0.18, 0.05, 0.08, 0.09, 0.05]
COUNTRIES = ["US", "CA", "GB", "DE", "MX"]
RISKY = ["crypto", "gambling", "wire_transfer", "gift_cards"]


def generate_synthetic_transactions(n_customers: int = 300, days: int = 90,
                                    seed: int = 42, anomaly_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-07-01")

    cust_ids = [f"C{i:04d}" for i in range(n_customers)]
    home = rng.choice(COUNTRIES, n_customers, p=[0.80, 0.06, 0.05, 0.05, 0.04])
    med_amt = rng.lognormal(3.4, 0.4, n_customers)          # typical ~ $30
    rate = rng.uniform(0.2, 1.0, n_customers)               # txns per day

    hour_w = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 8, 8, 9, 10, 9, 8, 8, 8, 9, 9, 8, 6, 4, 3, 2], float)
    hour_p = hour_w / hour_w.sum()

    frames = []
    for i, cid in enumerate(cust_ids):
        n = rng.poisson(rate[i] * days)
        if n == 0:
            continue
        day = rng.integers(0, days, n)
        hr = rng.choice(24, n, p=hour_p)
        mn = rng.integers(0, 60, n)
        ts = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hr, unit="h") + pd.to_timedelta(mn, unit="m")
        frames.append(pd.DataFrame({
            "customer_id": cid,
            "timestamp": ts,
            "amount": np.round(med_amt[i] * rng.lognormal(0, 0.5, n), 2),
            "merchant_category": rng.choice(CATEGORIES, n, p=CAT_P),
            "country": home[i],
            "synthetic_fraud_label": 0,
        }))
    base = pd.concat(frames, ignore_index=True)

    # A few legitimate-but-large purchases so the rules are not trivially perfect
    n_benign = int(0.007 * len(base))
    idx = rng.choice(len(base), n_benign, replace=False)
    base.loc[idx, "amount"] = np.round(base.loc[idx, "amount"] * rng.uniform(4, 8, n_benign), 2)

    cust_idx = {c: i for i, c in enumerate(cust_ids)}
    rows: list[dict] = []
    target = int(anomaly_rate * len(base))
    while len(rows) < target:
        cid = cust_ids[rng.integers(0, n_customers)]
        i = cust_idx[cid]
        kind = rng.choice(["large", "burst", "foreign"])
        t0 = start + pd.Timedelta(minutes=int(rng.integers(0, days * 24 * 60)))
        if kind == "large":
            rows.append(dict(customer_id=cid, timestamp=t0, amount=round(med_amt[i] * rng.uniform(8, 20), 2),
                             merchant_category=str(rng.choice(["retail", "travel", "wire_transfer"])),
                             country=home[i], synthetic_fraud_label=1))
        elif kind == "burst":
            for j in range(int(rng.integers(6, 9))):
                rows.append(dict(customer_id=cid, timestamp=t0 + pd.Timedelta(minutes=2 * j),
                                 amount=round(med_amt[i] * rng.uniform(1, 3), 2),
                                 merchant_category=str(rng.choice(["retail", "gift_cards"])),
                                 country=home[i], synthetic_fraud_label=1))
        else:
            foreign = str(rng.choice([c for c in COUNTRIES if c != home[i]]))
            t0 = t0.normalize() + pd.Timedelta(hours=int(rng.integers(1, 5)), minutes=int(rng.integers(0, 60)))
            rows.append(dict(customer_id=cid, timestamp=t0, amount=round(med_amt[i] * rng.uniform(3, 8), 2),
                             merchant_category=str(rng.choice(RISKY)), country=foreign, synthetic_fraud_label=1))

    df = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    d = generate_synthetic_transactions()
    d.to_csv("sample_transactions.csv", index=False)
    print(f"Wrote sample_transactions.csv with {len(d)} rows "
          f"({int(d.synthetic_fraud_label.sum())} synthetic anomalies)")
