from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from usdtry_risk.results import load_table

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["Overview", "Forecast vs naive", "Risk", "Hedging"]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.header[0].value


def test_full_hedge_matches_saved_backtest(app):
    app.sidebar.radio[0].set_value("Hedging").run()
    app.slider[0].set_value(100).run()
    worst = next(m for m in app.metric if m.label == "Worst 5% of months")
    expected = load_table("hedging_summary").loc["forward_100", "p5_pct"]
    assert worst.value == f"{expected:+.2f}%"
