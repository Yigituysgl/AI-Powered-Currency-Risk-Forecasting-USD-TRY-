"""USD/TRY risk and hedging dashboard for a USD exporter.

Reads the committed snapshot and the tables in reports/results; it never
downloads data or trains a model. Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Streamlit Cloud runs the file directly, without installing the package.
sys.path.insert(0, str(ROOT / "src"))

from usdtry_risk.data.prepare import largest_moves, load_snapshot, log_returns  # noqa: E402
from usdtry_risk.hedging.simulate import hedge_outcomes, summarize  # noqa: E402
from usdtry_risk.results import load_table  # noqa: E402

PRICE_COLOR = "#2a78d6"
SHOCK_COLOR = "#eb6834"
N_SHOCKS = 10
SIGNIFICANCE = 0.05

MODEL_NAMES = {
    "lstm": "LSTM",
    "naive": "Naive (tomorrow = today)",
    "drift_expanding": "Drift, average of all past days",
    "drift_252d": "Drift, average of the past year",
    "ewma": "EWMA",
    "garch": "GARCH(1,1)",
    "historical": "Historical",
}
STRATEGY_NAMES = {
    "no_hedge": "No hedge",
    "forward_50": "50% forward",
    "forward_100": "100% forward",
    "dynamic": "Dynamic (50% or 100%)",
    "forward_100_callmoney": "100% forward, call-money rate",
}


@st.cache_data
def prices() -> pd.Series:
    return load_snapshot()


@st.cache_data
def table(name: str, index_cols: int = 1) -> pd.DataFrame:
    return load_table(name, index_cols=index_cols)


def overview() -> None:
    p = prices()
    st.header("USD/TRY since 1999")
    st.write(
        f"Daily ECB reference rates, {p.index[0]:%d %b %Y} to {p.index[-1]:%d %b %Y} "
        f"({len(p):,} days). Orange points mark the {N_SHOCKS} largest one-day moves."
    )

    shock_frame = largest_moves(p, N_SHOCKS).reset_index()
    frame = p.rename("usd_try").rename_axis("date").reset_index()
    line = alt.Chart(frame).mark_line(color=PRICE_COLOR, strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("usd_try:Q", title="TRY per USD"),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("usd_try:Q", format=".4f")],
    )
    points = alt.Chart(shock_frame).mark_circle(color=SHOCK_COLOR, size=90, opacity=1).encode(
        x="date:T",
        y="usd_try:Q",
        tooltip=[
            alt.Tooltip("date:T", format="%d %b %Y"),
            alt.Tooltip("usd_try:Q", format=".4f"),
            alt.Tooltip("move_pct:Q", title="one-day move (%)", format="+.2f"),
        ],
    )
    st.altair_chart(line + points, width="stretch")

    st.subheader("Largest one-day moves")
    st.write("A negative move means the lira strengthened, which is a loss for an exporter holding USD.")
    st.dataframe(
        shock_frame.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
        column_config={
            "usd_try": st.column_config.NumberColumn("USD/TRY", format="%.4f"),
            "move_pct": st.column_config.NumberColumn("One-day move (%)", format="%+.2f"),
        },
        hide_index=True,
    )


def forecast() -> None:
    scores = table("forecast_scores")
    dm = table("forecast_dm")
    a = scores.attrs
    st.header("Forecast vs naive")
    st.info("Daily direction isn't forecastable.")
    st.write(
        f"Each model forecast the next day's rate on the same {a['n']:,} days "
        f"({a['start']} to {a['end']}), trained only on earlier data. "
        "Errors are on daily changes in percent; lower is better."
    )
    shown = scores[["mae_ret_pct", "rmse_ret_pct", "dir_acc"]].rename(index=MODEL_NAMES)
    st.dataframe(
        shown,
        column_config={
            "mae_ret_pct": st.column_config.NumberColumn("Average error (MAE, %)", format="%.4f"),
            "rmse_ret_pct": st.column_config.NumberColumn("RMSE (%)", format="%.4f"),
            "dir_acc": st.column_config.NumberColumn("Direction right", format="%.4f"),
        },
    )
    st.write(
        f"USD/TRY rose on {a['up_share']:.1%} of these days. The LSTM called the direction right on "
        f"{scores.loc['lstm', 'dir_acc']:.1%} of the days it predicted a move, so it does no better "
        "than always saying \"up\". Naive forecasts no change, so it has no direction."
    )

    st.subheader("Is the LSTM's difference real? (Diebold-Mariano test)")
    st.write(
        f"A p-value below {SIGNIFICANCE} means the difference in errors is unlikely to be luck. "
        "A negative statistic means the LSTM had the smaller errors."
    )
    st.dataframe(
        dm,
        column_config={
            "dm_abs": st.column_config.NumberColumn("Statistic, absolute errors", format="%.4f"),
            "p_abs": st.column_config.NumberColumn("p-value, absolute errors", format="%.4f"),
            "dm_sq": st.column_config.NumberColumn("Statistic, squared errors", format="%.4f"),
            "p_sq": st.column_config.NumberColumn("p-value, squared errors", format="%.4f"),
        },
    )
    for comparison, row in dm.iterrows():
        rival = MODEL_NAMES[comparison.removeprefix("lstm vs ")]
        parts = []
        for loss, tag in (("absolute", "abs"), ("squared", "sq")):
            p = row[f"p_{tag}"]
            if p < SIGNIFICANCE:
                side = "smaller" if row[f"dm_{tag}"] < 0 else "larger"
                parts.append(f"{loss} errors significantly {side} (p = {p:.4f})")
            else:
                parts.append(f"{loss} errors not significantly different (p = {p:.4f})")
        st.write(f"- LSTM against {rival}: {'; '.join(parts)}.")


def risk() -> None:
    current = table("var_current", index_cols=3)
    backtest = table("var_backtest", index_cols=2)
    a, b = current.attrs, backtest.attrs
    returns = log_returns(prices())
    st.header("Risk: how much could a USD 1,000,000 receivable lose?")
    st.write(
        f"Value at Risk (VaR) is a loss in TRY that should be exceeded only rarely: on 5% of days at 95% "
        f"confidence, on 1% at 99%. The exporter loses when USD/TRY falls. Figures are made on "
        f"{a['date']} at USD/TRY {a['usd_try']:.4f} for USD {a['exposure_usd']:,}."
    )
    view = current.reset_index()
    view["model"] = view["model"].map(MODEL_NAMES)
    view["confidence"] = view["confidence"].map("{:.0%}".format)
    for horizon, label in ((1, "Next day"), (21, "Next 21 trading days (about a month)")):
        st.subheader(label)
        st.dataframe(
            view[view["horizon_days"] == horizon].drop(columns="horizon_days"),
            column_config={
                "var_pct": st.column_config.NumberColumn("VaR (% of value)", format="%.4f"),
                "var_try": st.column_config.NumberColumn("VaR (TRY)", format="%,.0f"),
            },
            hide_index=True,
        )
    recent = returns.iloc[-500:].rolling(21).sum().dropna()
    st.write(
        "The models disagree, and the spread is the honest answer. A negative historical VaR means "
        f"USD/TRY rose in {(recent > 0).sum()} of the last {len(recent)} 21-day windows, so past "
        "data shows no monthly loss at that confidence. EWMA and GARCH assume no trend."
    )

    st.subheader("Did the 1-day VaR hold up in the past?")
    st.write(
        f"Each model's VaR, made each day from earlier data only, was checked against the next day's move "
        f"on {b['n']:,} days ({b['start']} to {b['end']}). A breach is a day the loss was larger than VaR. "
        f"Kupiec's test p-value below {SIGNIFICANCE} means the breach count is too far from what was expected."
    )
    bt = backtest.reset_index()
    bt["model"] = bt["model"].map(MODEL_NAMES)
    bt["confidence"] = bt["confidence"].map("{:.0%}".format)
    bt["mean_var"] = 100 * bt["mean_var"]
    bt["verdict"] = bt["kupiec_p"].map(lambda p: "consistent" if p >= SIGNIFICANCE else "rejected")
    st.dataframe(
        bt.drop(columns="kupiec_lr"),
        column_config={
            "expected": st.column_config.NumberColumn("Expected breaches", format="%.1f"),
            "breach_rate": st.column_config.NumberColumn("Breach rate", format="%.4f"),
            "kupiec_p": st.column_config.NumberColumn("Kupiec p-value", format="%.4f"),
            "mean_var": st.column_config.NumberColumn("Average VaR (%)", format="%.4f"),
        },
        hide_index=True,
    )
    st.write(
        "Too few breaches is also a failure: that VaR was larger than needed. "
        "The average VaR column shows what each model's caution costs."
    )


def hedging() -> None:
    receivables = table("hedging_receivables")
    summary = table("hedging_summary")
    a = summary.attrs
    st.header("Hedging: sell some USD forward?")
    st.write(
        f"Backtest of {a['n']} monthly receivables, booked {a['start']} to {a['end']}, each paid three "
        "months later. A forward fixes the TRY rate on the booking date; the rest is converted at the "
        "spot rate on payment."
    )
    amount = st.number_input("USD amount per month", min_value=1_000, value=1_000_000, step=100_000)
    ratio = st.slider("Hedge ratio (%)", 0, 100, 50, step=5)

    outcomes = hedge_outcomes(receivables, {"no_hedge": 0.0, "chosen": ratio / 100}, amount)
    result = summarize(outcomes, receivables, amount_usd=amount).loc["chosen"]
    st.caption("Results are in percent of the TRY value at the booking-date rate, which compares fairly across years.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average result", f"{result['mean_pct']:+.2f}%")
    c2.metric("Variability (std. dev.)", f"{result['std_pct']:.2f}%")
    c3.metric("Worst 5% of months", f"{result['p5_pct']:+.2f}%")
    st.write(
        f"In TRY for USD {amount:,.0f}: average {result['mean_try']:,.0f}, worst 5% of months "
        f"{result['p5_try']:,.0f} or less. TRY amounts mix years with very different exchange rates."
    )

    st.subheader("Strategy comparison (USD 1,000,000 per month)")
    shown = summary[["mean_pct", "std_pct", "p5_pct", "beat_baseline"]].rename(index=STRATEGY_NAMES)
    st.dataframe(
        shown,
        column_config={
            "mean_pct": st.column_config.NumberColumn("Average (%)", format="%+.4f"),
            "std_pct": st.column_config.NumberColumn("Variability (%)", format="%.4f"),
            "p5_pct": st.column_config.NumberColumn("Worst 5% (%)", format="%+.4f"),
            "beat_baseline": st.column_config.NumberColumn("Share of months beating no hedge", format="%.4f"),
        },
    )
    st.write(
        "Hedging more gives up some average result in exchange for less variability and a better worst "
        "case. The call-money row repeats the 100% forward with a market lira interest rate instead of "
        "the central bank's policy rate."
    )


PAGES = {"Overview": overview, "Forecast vs naive": forecast, "Risk": risk, "Hedging": hedging}

st.set_page_config(page_title="USD/TRY risk and hedging", layout="wide")
st.sidebar.title("USD/TRY exporter risk")
page = st.sidebar.radio("Page", list(PAGES))
PAGES[page]()
st.sidebar.caption(f"Data: ECB reference rates, snapshot {table('forecast_scores').attrs['snapshot']}.")
