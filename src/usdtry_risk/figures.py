"""Draw the README figures from the snapshot and the saved result tables.

Usage:
    python -m usdtry_risk.figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from usdtry_risk.data.prepare import largest_moves, load_snapshot  # noqa: E402
from usdtry_risk.hedging.simulate import hedge_outcomes, summarize  # noqa: E402
from usdtry_risk.results import RESULTS_DIR, load_table  # noqa: E402

FIGURES_DIR = RESULTS_DIR.parent / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.figsize": (9, 4.5),
        "figure.dpi": 120,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def price_history(prices: pd.Series) -> plt.Figure:
    moves = largest_moves(prices)
    fig, ax = plt.subplots()
    ax.plot(prices.index, prices, color=BLUE)
    ax.scatter(moves.index, moves["usd_try"], s=40, color=ORANGE, zorder=3, edgecolor="white", linewidth=1)
    ax.set_title("USD/TRY with the 10 largest one-day moves")
    ax.set_ylabel("TRY per USD")
    return fig


def forecast_error_by_year(folds: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(folds.index, folds["naive_mae"], color=ORANGE, marker="o", label="Naive (tomorrow = today)")
    ax.plot(folds.index, folds["lstm_mae"], color=BLUE, marker="o", label="LSTM")
    ax.set_title("Average daily forecast error by year: LSTM vs naive")
    ax.set_ylabel("Mean absolute error (%)")
    ax.set_ylim(bottom=0)
    ax.legend()
    return fig


def var_backtest(backtest: pd.DataFrame) -> plt.Figure:
    confidences = backtest.index.get_level_values("confidence").unique()
    fig, axes = plt.subplots(1, len(confidences), sharey=False)
    for ax, confidence in zip(axes, confidences):
        table = backtest.loc[confidence].rename(index={"ewma": "EWMA", "garch": "GARCH", "historical": "Historical"})
        ax.bar(table.index,100 * table["breach_rate"], color=BLUE, width=0.5)
        expected = 100 * (1 - confidence)
        ax.axhline(expected, color=INK, linestyle="--", linewidth=1)
        ax.text(-0.4, expected, "expected", ha="left", va="bottom", color=INK)
        ax.set_title(f"{confidence:.0%} VaR")
        ax.set_ylabel("Days with loss beyond VaR (%)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("1-day VaR backtest", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def hedge_tradeoff(receivables: pd.DataFrame) -> plt.Figure:
    ratios = [r / 100 for r in range(0, 101, 10)]
    outcomes = hedge_outcomes(receivables, {r: r for r in ratios})
    summary = summarize(outcomes, receivables, baseline=0.0)
    pct = [100 * r for r in ratios]
    fig, ax = plt.subplots()
    ax.plot(pct, summary["mean_pct"], color=BLUE, marker="o", label="Average result")
    ax.plot(pct, summary["p5_pct"], color=ORANGE, marker="o", label="Worst 5% of months")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("Hedge ratio trade-off, monthly USD receivables")
    ax.set_xlabel("Share sold forward (%)")
    ax.set_ylabel("TRY received vs value at booking (%)")
    ax.legend()
    return fig


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "price_history": price_history(load_snapshot()),
        "forecast_error_by_year": forecast_error_by_year(load_table("forecast_folds")),
        "var_backtest": var_backtest(load_table("var_backtest", index_cols=2)),
        "hedge_tradeoff": hedge_tradeoff(load_table("hedging_receivables")),
    }
    for name, fig in figures.items():
        path = FIGURES_DIR / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(path.relative_to(FIGURES_DIR.parents[1]))


if __name__ == "__main__":
    main()
