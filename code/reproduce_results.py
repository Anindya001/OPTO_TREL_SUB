"""Reproduce the principal numerical results from the archived data and draws.

This script generates numerical outputs only. It recomputes the central
quantities in the reliability analysis and writes plain CSV/JSON files to
``results/recomputed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis_core import (
    DRAWS_DIR,
    RESULTS_DIR,
    classical_rhat,
    common_shape_lrt,
    effective_sample_size,
    fit_curved_weibull,
    fit_lognormal_aft,
    fit_primary_mle,
    load_dataset,
    omission_subsets,
    profile_b10_interval,
    waic_dic,
    weibull_b10,
    write_json,
)

OUTPUT = RESULTS_DIR / "recomputed"



def reproduce_model_fits(data: pd.DataFrame) -> None:
    primary = fit_primary_mle(data)
    primary_loglik = float(primary["loglik"])
    n = len(data)
    primary.update(
        {
            "aic": -2.0 * primary_loglik + 2.0 * 3,
            "bic": -2.0 * primary_loglik + 3.0 * np.log(n),
        }
    )
    write_json(OUTPUT / "primary_fit.json", primary)
    write_json(OUTPUT / "curved_fit.json", fit_curved_weibull(data))
    write_json(OUTPUT / "lognormal_fit.json", fit_lognormal_aft(data))

    baseline = np.load(DRAWS_DIR / "baseline_chains.npy")
    write_json(OUTPUT / "information_criteria.json", waic_dic(baseline, data))


def reproduce_thermal_programme() -> None:
    profiles = []
    for excursion in (25, 50, 75, 100, 125, 150):
        ramp = excursion / 2.5
        dwell = 90.0 - ramp
        profiles.append(
            {
                "excursion_degC": excursion,
                "heating_min": ramp,
                "hot_dwell_min": dwell,
                "cooling_min": ramp,
                "cold_dwell_min": dwell,
                "period_min": 180.0,
            }
        )
    write_json(
        OUTPUT / "thermal_programme.json",
        {
            "source": "Specified set-points and timings; not measured temperature traces",
            "lower_set_point_degC": 0,
            "ramp_magnitude_degC_per_min": 2.5,
            "profiles": profiles,
        },
    )


def reproduce_omission_analysis(data: pd.DataFrame) -> None:
    rows = []
    for name, subset in omission_subsets(data):
        fit = fit_primary_mle(subset)
        theta = np.asarray(fit["theta"], dtype=float)
        rows.append(
            {
                "analysis": name,
                "n": len(subset),
                "k": theta[0],
                "beta1": theta[1],
                "beta0": theta[2],
                "B10_25": float(weibull_b10(theta, 25.0)),
                "B10_35": float(weibull_b10(theta, 35.0)),
                "loglik": fit["loglik"],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT / "omission.csv", index=False)


def reproduce_profile_interval(data: pd.DataFrame) -> None:
    write_json(OUTPUT / "profile_35.json", profile_b10_interval(data, 35.0))


def reproduce_common_shape(data: pd.DataFrame) -> None:
    fits, test = common_shape_lrt(data)
    fits.to_csv(OUTPUT / "group_shape_fits.csv", index=False)
    write_json(OUTPUT / "common_shape.json", test)


def reproduce_posterior_summaries() -> None:
    baseline = np.load(DRAWS_DIR / "baseline_chains.npy")
    pooled = baseline.reshape(-1, 3)

    parameter_rows = []
    for index, name in enumerate(("k", "beta1", "beta0")):
        q025, median, q975 = np.quantile(pooled[:, index], [0.025, 0.5, 0.975])
        parameter_rows.append(
            {
                "parameter": name,
                "q025": q025,
                "median": median,
                "q975": q975,
            }
        )
    pd.DataFrame(parameter_rows).to_csv(OUTPUT / "parameters.csv", index=False)

    diagnostic_rows = []
    for setting in ("baseline", "relaxed", "diffuse", "conservative"):
        chains = np.load(DRAWS_DIR / f"{setting}_chains.npy")
        rhat = classical_rhat(chains)
        for index, name in enumerate(("k", "beta1", "beta0")):
            diagnostic_rows.append(
                {
                    "setting": setting,
                    "parameter": name,
                    "rhat": rhat[index],
                    "ess": effective_sample_size(chains[:, :, index]),
                }
            )
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "mcmc_diagnostics.csv", index=False
    )

    life_rows = []
    for excursion in (20.0, 25.0, 35.0, 50.0, 75.0, 100.0, 125.0, 150.0):
        b10 = weibull_b10(pooled, excursion)
        q025, median, q975 = np.quantile(b10, [0.025, 0.5, 0.975])
        eta = np.exp(pooled[:, 2] - pooled[:, 1] * np.log(excursion))
        survival_5000 = np.mean(
            np.exp(-((5000.0 / eta) ** pooled[:, 0]))
        )
        life_rows.append(
            {
                "delta_T": excursion,
                "q025": q025,
                "median": median,
                "q975": q975,
                "survival_5000": survival_5000,
                "prob_B10_gt_5000": np.mean(b10 > 5000.0),
            }
        )
    pd.DataFrame(life_rows).to_csv(OUTPUT / "primary_life.csv", index=False)

    crossing_rows = []
    for mission in (5000.0, 10000.0):
        crossing = np.exp(
            (
                pooled[:, 2]
                + np.log(-np.log(0.9)) / pooled[:, 0]
                - np.log(mission)
            )
            / pooled[:, 1]
        )
        for probability in (0.50, 0.90, 0.95):
            crossing_rows.append(
                {
                    "mission_cycles": mission,
                    "posterior_probability": probability,
                    "crossing": np.quantile(crossing, 1.0 - probability),
                }
            )
    pd.DataFrame(crossing_rows).to_csv(OUTPUT / "crossings.csv", index=False)

    eta75 = np.exp(pooled[:, 2] - pooled[:, 1] * np.log(75.0))
    failure_probability = 1.0 - np.exp(
        -((1066.6666666666667 / eta75) ** pooled[:, 0])
    )
    pmf = np.mean(
        stats.binom.pmf(np.arange(9)[:, None], 8, failure_probability[None, :]),
        axis=1,
    )
    write_json(
        OUTPUT / "prediction75_full.json",
        {"count": list(range(9)), "pmf": pmf.tolist()},
    )


def reproduce_prior_sensitivity() -> None:
    rows = []
    for setting in ("baseline", "relaxed", "diffuse", "conservative"):
        draws = np.load(DRAWS_DIR / f"{setting}_chains.npy")
        for excursion in (20.0, 25.0, 35.0):
            values = weibull_b10(draws, excursion).reshape(-1)
            q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
            rows.append(
                {
                    "setting": setting,
                    "delta_T": excursion,
                    "q025": q025,
                    "median": median,
                    "q975": q975,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "prior_sensitivity.csv", index=False)


def reproduce_model_form_summaries() -> None:
    baseline = np.load(DRAWS_DIR / "baseline_chains.npy").reshape(-1, 3)
    curved = np.load(DRAWS_DIR / "curved_laplace.npy")
    lognormal = np.load(DRAWS_DIR / "lognormal_laplace.npy")
    xbar = np.mean(np.log([25.0, 50.0, 75.0, 100.0, 125.0, 150.0]))

    rows = []
    for excursion in (20.0, 25.0, 35.0, 50.0, 75.0, 100.0, 125.0, 150.0):
        primary = weibull_b10(baseline, excursion)
        curved_b10 = np.exp(
            curved[:, 2]
            - curved[:, 1] * np.log(excursion)
            + curved[:, 3] * (np.log(excursion) - xbar) ** 2
            + np.log(-np.log(0.9)) / curved[:, 0]
        )
        lognormal_b10 = np.exp(
            lognormal[:, 2]
            - lognormal[:, 1] * np.log(excursion)
            + lognormal[:, 0] * stats.norm.ppf(0.10)
        )

        for name, values in (
            ("Weibull log-linear", primary),
            ("Weibull curved", curved_b10),
            ("Lognormal", lognormal_b10),
        ):
            q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
            rows.append(
                {
                    "model": name,
                    "delta_T": excursion,
                    "q025": q025,
                    "median": median,
                    "q975": q975,
                }
            )

    pd.DataFrame(rows).to_csv(OUTPUT / "model_life.csv", index=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset()

    reproduce_model_fits(data)
    reproduce_thermal_programme()
    reproduce_omission_analysis(data)
    reproduce_profile_interval(data)
    reproduce_common_shape(data)
    reproduce_posterior_summaries()
    reproduce_prior_sensitivity()
    reproduce_model_form_summaries()

    print(f"Recomputed results written to: {OUTPUT}")


if __name__ == "__main__":
    main()
