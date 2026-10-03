"""Standardisation whose parameters come only from the data passed to ``fit``."""
from __future__ import annotations

import pandas as pd


class StandardScaler:
    """Subtract the mean and divide by the standard deviation of the fit data."""

    def fit(self, x: pd.Series) -> StandardScaler:
        self.mean_ = float(x.mean())
        self.std_ = float(x.std(ddof=0))
        if not self.std_ > 0:
            raise ValueError("cannot scale a constant series")
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, z):
        return z * self.std_ + self.mean_
