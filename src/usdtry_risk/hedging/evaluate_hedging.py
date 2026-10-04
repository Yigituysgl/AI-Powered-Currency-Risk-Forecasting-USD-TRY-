"""Backtest hedging strategies for monthly USD 1,000,000 receivables due in 3 months.

Usage:
    python -m usdtry_risk.hedging.evaluate_hedging [--start 2010-06]
"""
from __future__ import annotations

import argparse

import pandas as pd

from usdtry_risk.data.prepare import load_snapshot, log_returns
from usdtry_risk.data.rates import TRY_CALL_MONEY, TRY_POLICY, USD_TBILL, rate_series
from usdtry_risk.hedging.forward import monthly_available_from
from usdtry_risk.hedging.simulate import (
    AMOUNT_USD,
    build_receivables,
    decision_dates,
    dynamic_hedge_ratio,
    hedge_outcomes,
    summarize,
)
from usdtry_risk.results import save_table


def policy_vs_call_money(policy: pd.Series, call_money: pd.Series, start: str) -> pd.DataFrame:
    """Monthly call money rate beside the TCMB policy rate in effect at each month end."""
    monthly = call_money[call_money.index >= start]
    month_ends = monthly.index + pd.offsets.MonthEnd(0)
    in_effect = policy.reindex(policy.index.union(month_ends)).ffill().reindex(month_ends)
    return pd.DataFrame(
        {
            "call_money": monthly.to_numpy(),
            "policy_at_month_end": in_effect.to_numpy(),
            "gap": monthly.to_numpy() - in_effect.to_numpy(),
        },
        index=monthly.index.to_period("M"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2010-06")
    parser.add_argument("--save", action="store_true", help="write tables to reports/results")
    args = parser.parse_args()

    prices = load_snapshot()
    usd = rate_series(USD_TBILL)
    policy = rate_series(TRY_POLICY)
    call_money = rate_series(TRY_CALL_MONEY)
    dates = decision_dates(prices, args.start)

    receivables = build_receivables(prices, usd, policy, dates)
    receivables_cm = build_receivables(prices, usd, monthly_available_from(call_money), dates)
    if not receivables.index.equals(receivables_cm.index):
        raise RuntimeError("both rate sources must cover the same receivables")

    dynamic = dynamic_hedge_ratio(log_returns(prices), receivables.index)
    outcomes = hedge_outcomes(
        receivables,
        {"no_hedge": 0.0, "forward_50": 0.5, "forward_100": 1.0, "dynamic": dynamic},
    )
    outcomes["forward_100_callmoney"] = AMOUNT_USD * receivables_cm["forward"]
    result = summarize(outcomes, receivables)
    a = result.attrs
    if args.save:
        save_table(result.rename_axis("strategy"), "hedging_summary")
        # Outcomes are linear in amount and hedge ratio, so these columns let the
        # dashboard re-price any amount and ratio without the rate snapshots.
        monthly = receivables.assign(dynamic_ratio=dynamic)
        monthly.attrs.update(result.attrs)
        save_table(monthly, "hedging_receivables")

    with pd.option_context(
        "display.float_format", "{:,.4f}".format, "display.width", 160, "display.max_columns", None
    ):
        print(
            f"{a['n']} receivables, decisions {a['start']:%Y-%m-%d} to {a['end']:%Y-%m-%d}, "
            f"last maturity {receivables['maturity'].max():%Y-%m-%d}"
        )
        print(result)
        print(f"\nDynamic rule fully hedged in {int((dynamic == 1.0).sum())} of {len(dynamic)} months")
        print(
            f"Forward above decision spot in {int((receivables['forward'] > receivables['spot']).sum())} "
            f"of {len(receivables)}; decision spot {receivables['spot'].iloc[0]:.4f} first, "
            f"{receivables['spot'].iloc[-1]:.4f} last"
        )
        print(
            "Mean rates at decision (% per year): "
            f"TRY policy {receivables['r_try'].mean():.4f}, "
            f"TRY call money {receivables_cm['r_try'].mean():.4f}, "
            f"USD T-bill {receivables['r_usd'].mean():.4f}"
        )
        print(f"\nTCMB table: last change {policy.index[-1]:%Y-%m-%d} to {policy.iloc[-1]:.2f}%")
        print("Call money vs TCMB policy rate, monthly:")
        print(policy_vs_call_money(policy, call_money, "2025-01"))


if __name__ == "__main__":
    main()
