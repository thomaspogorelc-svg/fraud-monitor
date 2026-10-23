"""Streamlit UI for the Fraud & Transaction Monitoring Prototype."""
import pandas as pd
import plotly.express as px
import streamlit as st

from data_gen import generate_synthetic_transactions
from fraud import DataError, RuleConfig, evaluate, flag, score_transactions, validate_data

st.set_page_config(page_title="Fraud Monitoring Prototype", page_icon="🛡️", layout="wide")
st.title("🛡️ Transaction Fraud Monitoring Prototype")
st.caption("Synthetic data only. Alerts indicate anomalies, not confirmed fraud.")


@st.cache_data(show_spinner=False)
def _synthetic(n_customers: int, days: int, seed: int, rate: float) -> pd.DataFrame:
    return generate_synthetic_transactions(n_customers, days, seed, rate)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Generate synthetic data", "Upload CSV"])
    raw = None
    if source == "Generate synthetic data":
        n_cust = st.slider("Customers", 50, 1000, 300, 50)
        days = st.slider("Days of history", 30, 180, 90, 10)
        seed = st.number_input("Random seed", 0, 99999, 42)
        rate = st.slider("Injected anomaly rate", 0.005, 0.10, 0.02, 0.005, format="%.3f")
        raw = _synthetic(n_cust, days, int(seed), rate)
    else:
        up = st.file_uploader("CSV with columns: txn_id, customer_id, timestamp, amount, merchant_category, country",
                              type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:  # unreadable / malformed file
                st.error(f"Could not read the CSV: {e}")

    st.header("2. Risk threshold")
    threshold = st.slider("Flag transactions with score ≥", 0, 100, 50, 5)

    st.header("3. Rule parameters")
    z_thr = st.slider("Amount robust z-score trigger", 1.0, 10.0, 3.5, 0.5)
    vel_win = st.slider("Velocity window (minutes)", 5, 240, 60, 5)
    vel_cnt = st.slider("Txns in window to trigger", 2, 15, 5)
    night = st.slider("Unusual-hour window (hours)", 0, 23, (0, 5))
    large_mult = st.slider("'Large' = × customer median", 1.5, 10.0, 3.0, 0.5)

    st.header("4. Rule weights (max points)")
    w_amt = st.slider("Amount outlier", 0, 60, 45)
    w_vel = st.slider("Velocity", 0, 60, 40)
    w_geo = st.slider("Foreign country", 0, 60, 25)
    w_nt = st.slider("Unusual hour", 0, 60, 15)
    w_cat = st.slider("High-risk category", 0, 60, 25)

if raw is None:
    st.info("Upload a CSV in the sidebar, or switch to synthetic data.")
    st.stop()

try:
    clean, warns = validate_data(raw)
except DataError as e:
    st.error(f"Data problem: {e}")
    st.stop()
for w in warns:
    st.warning(w)

cfg = RuleConfig(amount_z_threshold=z_thr, velocity_window_min=vel_win, velocity_count=vel_cnt,
                 night_start=night[0], night_end=night[1], large_multiple=large_mult,
                 w_amount=w_amt, w_velocity=w_vel, w_geo=w_geo, w_night=w_nt, w_category=w_cat)

try:
    scored = score_transactions(clean, cfg)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

flagged = flag(scored, threshold)

# ---------------- Metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(scored):,}")
c2.metric("Customers", f"{scored['customer_id'].nunique():,}")
c3.metric("Flagged", f"{len(flagged):,}", f"{len(flagged) / len(scored):.2%} of total", delta_color="off")
c4.metric("Flagged amount", f"${flagged['amount'].sum():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Patterns", "🚩 Flagged transactions", "🎯 Evaluation", "📘 Methodology"])

with tab1:
    a, b = st.columns(2)
    fig = px.histogram(scored, x="amount", nbins=60, log_y=True, title="Transaction amounts (log count)")
    a.plotly_chart(fig, use_container_width=True)
    daily = scored.set_index("timestamp").resample("D").agg(count=("amount", "size"), total=("amount", "sum")).reset_index()
    fig = px.line(daily, x="timestamp", y="count", title="Daily transaction count")
    b.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    per_cust = scored.groupby("customer_id").size().reset_index(name="txns")
    c.plotly_chart(px.histogram(per_cust, x="txns", nbins=40, title="Transactions per customer (frequency)"),
                   use_container_width=True)
    hourly = scored.groupby("hour").size().reset_index(name="txns")
    d.plotly_chart(px.bar(hourly, x="hour", y="txns", title="Transactions by hour of day"), use_container_width=True)
    fig = px.histogram(scored, x="risk_score", nbins=40, log_y=True, title="Risk score distribution")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="threshold")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if flagged.empty:
        st.success("No transactions meet the current threshold. Try lowering it.")
    else:
        show = ["txn_id", "customer_id", "timestamp", "amount", "merchant_category", "country", "risk_score", "explanation"]
        st.dataframe(flagged[show], use_container_width=True, hide_index=True)
        st.download_button("Download flagged CSV", flagged[show].to_csv(index=False), "flagged_transactions.csv", "text/csv")
        st.subheader("Why was it flagged?")
        pick = st.selectbox("Transaction", flagged["txn_id"].tolist())
        row = flagged[flagged["txn_id"] == pick].iloc[0]
        st.markdown(f"**Risk score: {row['risk_score']} / 100** — customer `{row['customer_id']}`, "
                    f"${row['amount']:,.2f}, {row['merchant_category']}, {row['country']}")
        for part in row["explanation"].split("; "):
            st.markdown(f"- {part}")
        pts = pd.DataFrame({
            "rule": ["Amount outlier", "Velocity", "Foreign country", "Unusual hour", "High-risk category"],
            "points": [row["pts_amount"], row["pts_velocity"], row["pts_geo"], row["pts_night"], row["pts_category"]],
        })
        st.plotly_chart(px.bar(pts, x="points", y="rule", orientation="h", title="Points contributed by each rule"),
                        use_container_width=True)

with tab3:
    ev = evaluate(scored, threshold)
    if ev is None:
        st.info("No `synthetic_fraud_label` column in this dataset, so evaluation is unavailable.")
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision", f"{ev['precision']:.1%}")
        e2.metric("Recall", f"{ev['recall']:.1%}")
        e3.metric("F1", f"{ev['f1']:.2f}")
        e4.metric("TP / FP / FN", f"{ev['tp']} / {ev['fp']} / {ev['fn']}")
        sweep = pd.DataFrame([{"threshold": t, **evaluate(scored, t)} for t in range(5, 101, 5)])
        long = sweep.melt("threshold", ["precision", "recall", "f1"], var_name="metric")
        st.plotly_chart(px.line(long, x="threshold", y="value", color="metric",
                                title="Precision / recall trade-off by threshold"), use_container_width=True)
        st.caption("Labels come from the synthetic generator's injected anomalies, so results are illustrative only.")

with tab4:
    st.markdown("""
**Risk score (0–100)** = sum of points from five transparent rules, capped at 100 (default max points: 45/40/25/15/25, so usually two or more rules must fire together to reach the default threshold of 50). Each rule yields an intensity
(0, 0.5 or 1) multiplied by its weight (editable in the sidebar).

| Rule | Trigger |
|---|---|
| Amount outlier | Robust z-score vs the customer's own median/MAD ≥ threshold (half-threshold = half points) |
| Velocity | ≥ N transactions by the same customer within the time window |
| Foreign country | Country differs from the customer's most common country |
| Unusual hour | Transaction hour falls in the chosen window |
| High-risk category | Crypto, gambling, wire transfer or gift cards (full points if also a large amount) |

**Limitations:** baselines use each customer's full history (including the transaction being scored); rules are
hand-set rather than learned; thresholds trade false positives against missed fraud; data is synthetic.
Alerts are anomalies for human review, not confirmed fraud.
""")
