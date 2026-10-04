import pandas as pd
import pytest

from usdtry_risk.hedging.forward import cip_forward, monthly_available_from, rate_known_before


def test_cip_forward_hand_computed():
    # 40% TRY and 5% USD over 91 days.
    t = 91 / 365
    assert cip_forward(30.0, 40.0, 5.0, 91) == pytest.approx(30.0 * (1 + 0.40 * t) / (1 + 0.05 * t))


def test_forward_premium_follows_the_rate_gap():
    assert cip_forward(30.0, 40.0, 5.0, 91) > 30.0
    assert cip_forward(30.0, 5.0, 5.0, 91) == pytest.approx(30.0)
    assert cip_forward(30.0, 2.0, 5.0, 91) < 30.0


def test_rate_known_before_excludes_the_same_day():
    rates = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert rate_known_before(rates, pd.Timestamp("2024-01-04")) == 2.0
    assert rate_known_before(rates, pd.Timestamp("2024-02-01")) == 3.0
    with pytest.raises(ValueError, match="known before"):
        rate_known_before(rates, pd.Timestamp("2024-01-02"))


def test_monthly_value_is_used_only_after_the_lag():
    monthly = pd.Series([10.0, 20.0], index=pd.to_datetime(["2026-06-01", "2026-07-01"]), name="m")
    available = monthly_available_from(monthly)
    assert list(available.index) == list(pd.to_datetime(["2026-09-01", "2026-10-01"]))
    assert rate_known_before(available, pd.Timestamp("2026-09-30")) == 10.0
