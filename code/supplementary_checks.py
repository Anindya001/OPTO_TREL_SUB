"""Supplementary numerical checks derived from the retained data and draws.

The script reproduces three descriptive quantities used to interpret the
results: the 100 °C group-only Weibull summary, an order-statistic dispersion
check, and in-sample inclusion of observed failures in the retained lifetime
bands.  These are descriptive checks, not additional fitted models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betaln

from analysis_core import DATA_FILE, DRAWS_DIR, RESULTS_DIR, load_dataset, write_json


def group_100_summary(data: pd.DataFrame) -> None:
    times = data.loc[data["delta_T_C"] == 100.0, "cycles"].to_numpy(dtype=float)
    k, _, eta = stats.weibull_min.fit(times, floc=0.0)
    percentiles_hours = 3.0 * stats.weibull_min.ppf(
        [0.025, 0.5, 0.975], k, scale=eta
    )
    write_json(
        RESULTS_DIR / "group100.json",
        {
            "k": float(k),
            "eta_cycles": float(eta),
            "percentiles_hours": percentiles_hours.tolist(),
        },
    )


def order_statistic_check() -> None:
    observed_ratio = 2818.0 / 720.0
    rows = []

    for shape in (2.791, 2.988):
        def survival(ratio: float) -> float:
            a = ratio**shape - 1.0
            if a <= 0.0:
                return 1.0
            return float(-np.expm1(np.log(8.0 / a) + betaln(8.0 / a, 8.0)))

        expected_ratio = 1.0 + quad(
            survival, 1.0, 100000.0, epsabs=1.0e-5, limit=200
        )[0]
        rows.append(
            {
                "shape": shape,
                "expected_ratio": expected_ratio,
                "probability_ratio_at_least_observed": survival(observed_ratio),
            }
        )

    write_json(
        RESULTS_DIR / "order_statistic_check.json",
        {
            "n": 8,
            "observed_ratio": observed_ratio,
            "method": "Exact order-statistic CDF with numerical tail integration",
            "checks": rows,
        },
    )


def lifetime_bands(draws: np.ndarray) -> pd.DataFrame:
    pooled = draws.reshape(-1, 3)
    grid = np.linspace(20.0, 150.0, 160)
    rows = []

    for excursion in grid:
        k = pooled[:, 0]
        log_scale = pooled[:, 2] - pooled[:, 1] * np.log(excursion)
        median_life = np.exp(log_scale + np.log(np.log(2.0)) / k)
        median_q025, median_q50, median_q975 = np.quantile(
            median_life, [0.025, 0.5, 0.975]
        )

        def mixture_cdf(log_life: float) -> float:
            power = np.clip(k * (log_life - log_scale), -745.0, 40.0)
            return float(np.mean(-np.expm1(-np.exp(power))))

        lower = np.exp(
            brentq(
                lambda value: mixture_cdf(value) - 0.025,
                float(log_scale.min() - 15.0),
                float(log_scale.max() + 15.0),
            )
        )
        upper = np.exp(
            brentq(
                lambda value: mixture_cdf(value) - 0.975,
                float(log_scale.min() - 15.0),
                float(log_scale.max() + 15.0),
            )
        )
        rows.append(
            {
                "delta_T": excursion,
                "median_q025": median_q025,
                "median_q50": median_q50,
                "median_q975": median_q975,
                "new_unit_q025": lower,
                "new_unit_q975": upper,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "lifetime_bands.csv", index=False)
    return table


def predictive_inclusion(data: pd.DataFrame, draws: np.ndarray, bands: pd.DataFrame) -> None:
    pooled = draws.reshape(-1, 3)
    rows = []

    for excursion, failures in data.loc[data["event"] == 1].groupby("delta_T_C"):
        cycles = failures["cycles"].to_numpy(dtype=float)
        k = pooled[:, 0]
        log_scale = pooled[:, 2] - pooled[:, 1] * np.log(excursion)
        median_life = np.exp(log_scale + np.log(np.log(2.0)) / k)
        credible = np.quantile(median_life, [0.025, 0.975])

        def mixture_cdf(log_life: float) -> float:
            power = np.clip(k * (log_life - log_scale), -745.0, 40.0)
            return float(np.mean(-np.expm1(-np.exp(power))))

        predictive = np.array(
            [
                np.exp(
                    brentq(
                        lambda value: mixture_cdf(value) - probability,
                        float(log_scale.min() - 15.0),
                        float(log_scale.max() + 15.0),
                    )
                )
                for probability in (0.025, 0.975)
            ]
        )

        plotted_predictive = np.array(
            [
                np.interp(excursion, bands["delta_T"], bands[column])
                for column in ("new_unit_q025", "new_unit_q975")
            ]
        )
        plotted_credible = np.array(
            [
                np.interp(excursion, bands["delta_T"], bands[column])
                for column in ("median_q025", "median_q975")
            ]
        )

        def count(bounds: np.ndarray) -> int:
            return int(np.sum((cycles >= bounds[0]) & (cycles <= bounds[1])))

        rows.append(
            {
                "delta_T_C": float(excursion),
                "observed_failures": len(cycles),
                "inside_predictive": count(predictive),
                "inside_median_credible": count(credible),
                "inside_plotted_predictive": count(plotted_predictive),
                "inside_plotted_median_credible": count(plotted_credible),
            }
        )

    totals = {
        key: int(sum(row[key] for row in rows))
        for key in (
            "observed_failures",
            "inside_predictive",
            "inside_median_credible",
            "inside_plotted_predictive",
            "inside_plotted_median_credible",
        )
    }

    write_json(
        RESULTS_DIR / "predictive_band_inclusion.json",
        {
            "scope": "In-sample inclusion only; not out-of-sample calibration.",
            "source_sha256": {
                "data": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                "baseline_draws": hashlib.sha256(
                    (DRAWS_DIR / "baseline_chains.npy").read_bytes()
                ).hexdigest(),
            },
            "stress_groups": rows,
            "totals": totals,
        },
    )


def main() -> None:
    data = load_dataset()
    draws = np.load(DRAWS_DIR / "baseline_chains.npy")
    group_100_summary(data)
    order_statistic_check()
    bands = lifetime_bands(draws)
    predictive_inclusion(data, draws, bands)
    print("Supplementary numerical checks completed.")


if __name__ == "__main__":
    main()
