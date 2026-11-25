"""Multiple linear regression model for the prevalence of undernourishment.

Mirrors the analysis in ``Results.ipynb``: four population-normalised
indicators are standardised on the training split and fitted with batch
gradient descent (1,500 iterations, learning rate 0.01) against the
prevalence of undernourishment. Unlike the notebook, the test split is
standardised with the training means and standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "DDW data.csv"

TARGET_COLUMN = "Prevalence of undernourishment (% of population)"

RANDOM_STATE = 9022
TEST_SIZE = 0.3
ITERATIONS = 1500
ALPHA = 0.01


@dataclass(frozen=True)
class Feature:
    key: str
    column: str
    label: str
    short: str
    unit: str
    hint: str
    kind: str  # "gdp" | "index" | "percent"


FEATURES = (
    Feature(
        key="gdp",
        column="Gross domestic product per capita, PPP, (constant 2017 international $)",
        label="GDP per capita, PPP",
        short="GDP / capita",
        unit="constant 2017 intl $",
        hint="Purchasing-power adjusted economic output per person.",
        kind="gdp",
    ),
    Feature(
        key="stability",
        column="Political stability and absence of violence/terrorism (index)",
        label="Political stability",
        short="Stability",
        unit="index (−2.5 to 2.5)",
        hint="Perceived likelihood of political instability and politically motivated violence.",
        kind="index",
    ),
    Feature(
        key="sanitation",
        column="Percentage of population using at least basic sanitation services (percent)",
        label="Basic sanitation access",
        short="Sanitation",
        unit="% of population",
        hint="Share of the population using at least basic sanitation services.",
        kind="percent",
    ),
    Feature(
        key="clean_fuels",
        column="Access to clean fuels and technologies for cooking (% of population)",
        label="Clean cooking access",
        short="Clean fuels",
        unit="% of population",
        hint="Share of the population cooking with clean fuels and technologies.",
        kind="percent",
    ),
)


def _bounds(feature: Feature, values: np.ndarray) -> tuple[float, float, float]:
    """Slider bounds and step for one indicator."""
    if feature.kind == "gdp":
        low = float(np.floor(values.min() / 1000) * 1000)
        high = float(np.ceil(values.max() / 1000) * 1000)
        return low, high, 100.0
    if feature.kind == "index":
        return -2.5, 2.5, 0.01
    return 0.0, 100.0, 0.1


def _scores(y: np.ndarray, pred: np.ndarray, n_features: int) -> dict[str, float]:
    residuals = y - pred
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    n = len(y)
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
    return {
        "r2": round(r2, 4),
        "adjusted_r2": round(adjusted, 4),
        "rmse": round(float(np.sqrt(ss_res / n)), 3),
    }


class UndernourishmentModel:
    def __init__(self, data_file: Path = DATA_FILE) -> None:
        frame = pd.read_csv(data_file, encoding="utf-8-sig")
        frame.columns = [name.strip() for name in frame.columns]
        self.countries = frame["Country Name"].astype(str).tolist()
        self.values = frame[[feature.column for feature in FEATURES]].to_numpy(dtype=float)
        self.target = frame[TARGET_COLUMN].to_numpy(dtype=float)
        self._fit()

    def _fit(self) -> None:
        n = len(self.target)
        rng = np.random.RandomState(RANDOM_STATE)
        test_index = np.sort(rng.choice(n, size=int(n * TEST_SIZE), replace=False))
        train_index = np.setdiff1d(np.arange(n), test_index)

        self.means = self.values[train_index].mean(axis=0)
        self.stds = self.values[train_index].std(axis=0, ddof=1)
        standardised = (self.values - self.means) / self.stds

        X_train = np.hstack([np.ones((len(train_index), 1)), standardised[train_index]])
        y_train = self.target[train_index].reshape(-1, 1)

        beta = np.zeros((X_train.shape[1], 1))
        for _ in range(ITERATIONS):
            beta -= ALPHA / len(X_train) * X_train.T @ (X_train @ beta - y_train)

        self.beta_std = beta.ravel()
        self.coef_std = self.beta_std[1:]
        self.coef = self.coef_std / self.stds
        self.intercept = float(self.beta_std[0] - np.sum(self.coef_std * self.means / self.stds))

        X_test = np.hstack([np.ones((len(test_index), 1)), standardised[test_index]])
        self.metrics = {
            "test": _scores(self.target[test_index], (X_test @ beta).ravel(), len(FEATURES)),
            "train": _scores(self.target[train_index], (X_train @ beta).ravel(), len(FEATURES)),
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
        }

        self.feature_info = []
        for i, feature in enumerate(FEATURES):
            low, high, step = _bounds(feature, self.values[:, i])
            default = float(np.clip(round(float(self.means[i]) / step) * step, low, high))
            self.feature_info.append(
                {
                    "key": feature.key,
                    "label": feature.label,
                    "short": feature.short,
                    "unit": feature.unit,
                    "hint": feature.hint,
                    "min": low,
                    "max": high,
                    "step": step,
                    "default": round(default, 2),
                    "mean": round(float(self.means[i]), 2),
                    "std": round(float(self.stds[i]), 2),
                    "coef": round(float(self.coef[i]), 4),
                    "coef_std": round(float(self.coef_std[i]), 3),
                }
            )

        self.defaults = {info["key"]: info["default"] for info in self.feature_info}
        self.country_rows = [
            {
                "name": name,
                "actual": round(float(self.target[i]), 1),
                "predicted": round(self.predict(self._values_dict(i)), 1),
                "values": self._values_dict(i),
            }
            for i, name in enumerate(self.countries)
        ]

    def _values_dict(self, index: int) -> dict[str, float]:
        return {
            feature.key: round(float(self.values[index, i]), 2)
            for i, feature in enumerate(FEATURES)
        }

    def predict(self, values: dict[str, float]) -> float:
        raw = self.intercept + sum(
            float(self.coef[i]) * float(values[feature.key])
            for i, feature in enumerate(FEATURES)
        )
        return float(np.clip(raw, 0.0, 100.0))

    def snapshot(self) -> dict:
        return {
            "target": {
                "label": "Prevalence of undernourishment",
                "unit": "% of population",
            },
            "metrics": {
                **self.metrics,
                "n_countries": len(self.target),
                "iterations": ITERATIONS,
                "alpha": ALPHA,
                "random_state": RANDOM_STATE,
            },
            "features": self.feature_info,
            "countries": self.country_rows,
            "defaults": self.defaults,
            "default_prediction": round(self.predict(self.defaults), 1),
        }


_model: UndernourishmentModel | None = None


def get_model() -> UndernourishmentModel:
    global _model
    if _model is None:
        _model = UndernourishmentModel()
    return _model
