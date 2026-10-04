"""Core numerical routines for the optocoupler reliability analysis.

The functions in this module are deliberately small and explicit.  They cover
only the calculations needed to reproduce and verify the numerical results
shipped with this archive:

* exact right-censored Weibull likelihood for the primary log-linear model;
* matched maximum-likelihood omission fits;
* ordinary profile-likelihood interval for B10 at a selected excursion;
* complete-failure common-shape likelihood-ratio diagnostic;
* posterior summaries from the retained production chains; and
* simple MCMC diagnostics used in the archived results.

Times are expressed in thermal cycles.  One cycle is three hours in the test
programme.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "optocoupler_ttf_unit_level.csv"
DRAWS_DIR = ROOT / "draws"
RESULTS_DIR = ROOT / "results"

CYCLE_HOURS = 3.0
B10_FAILURE_PROBABILITY = 0.10


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load and validate the unit-level lifetime dataset.

    The archived lifetime variable is stored in hours.  The analysis uses the
    exact conversion ``cycles = ttf_hours / 3``; the rounded helper column in
    the CSV is not used for fitting.
    """

    data = pd.read_csv(path).copy()
    required = {"delta_T_C", "ttf_hours", "event"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required data columns: {sorted(missing)}")

    data["cycles"] = data["ttf_hours"].astype(float) / CYCLE_HOURS
    data["delta_T_C"] = data["delta_T_C"].astype(float)
    data["event"] = data["event"].astype(int)

    if not set(data["event"].unique()).issubset({0, 1}):
        raise ValueError("event must contain only 0 (right-censored) or 1 (failed)")
    if (data["cycles"] <= 0).any() or (data["delta_T_C"] <= 0).any():
        raise ValueError("All durations and thermal excursions must be positive")

    return data


def log_eta(delta_t: np.ndarray, beta1: float, beta0: float) -> np.ndarray:
    """Log characteristic life for the primary log-linear stress relation."""

    return beta0 - beta1 * np.log(delta_t)


def weibull_log_likelihood(theta: Iterable[float], data: pd.DataFrame) -> float:
    """Exact right-censored Weibull log-likelihood for the primary model.

    ``theta = (k, beta1, beta0)`` where ``k`` is the Weibull shape and
    ``eta(d) = exp(beta0 - beta1 log(d))`` is the characteristic life.
    """

    k, beta1, beta0 = map(float, theta)
    if k <= 1.0 or beta1 <= 0.0:
        return -np.inf

    t = data["cycles"].to_numpy(dtype=float)
    d = data["delta_T_C"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=float)

    log_scale = log_eta(d, beta1, beta0)
    log_ratio = np.log(t) - log_scale
    cumulative_hazard = np.exp(np.clip(k * log_ratio, -700.0, 700.0))

    contribution = (
        event * (np.log(k) + (k - 1.0) * log_ratio - log_scale)
        - cumulative_hazard
    )
    return float(np.sum(contribution))


def fit_primary_mle(data: pd.DataFrame) -> dict[str, float | list[float]]:
    """Fit the primary censored Weibull model by maximum likelihood.

    Several deterministic starting values are used to reduce dependence on a
    single local starting point.  The bounds reflect the analysis assumptions
    ``k > 1`` and a positive stress slope.
    """

    bounds = [(1.0000001, 10.0), (1.0e-6, 10.0), (0.0, 40.0)]
    starts = (
        (3.0, 2.4, 17.6),
        (2.5, 2.0, 15.0),
        (4.0, 3.0, 20.0),
        (2.0, 1.0, 12.0),
    )

    best = None
    for start in starts:
        result = optimize.minimize(
            lambda x: -weibull_log_likelihood(x, data),
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 20000},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Primary Weibull optimisation failed")

    k, beta1, beta0 = best.x
    return {
        "theta": [float(k), float(beta1), float(beta0)],
        "loglik": float(-best.fun),
        "converged": bool(best.success),
    }


def weibull_b10(theta: np.ndarray | Iterable[float], delta_t: float) -> np.ndarray:
    """Return the Weibull B10 life at ``delta_t`` for one or many parameter sets."""

    values = np.asarray(theta, dtype=float)
    k = values[..., 0]
    beta1 = values[..., 1]
    beta0 = values[..., 2]
    constant = np.log(-np.log(1.0 - B10_FAILURE_PROBABILITY))
    return np.exp(beta0 - beta1 * np.log(delta_t) + constant / k)


def omission_subsets(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return the matched record subsets used in the omission analysis."""

    runout = data["event"] == 0
    low = data["delta_T_C"].isin([25.0, 50.0])
    transition = data["delta_T_C"] == 75.0

    return [
        ("Full likelihood", data),
        ("Omit 25/50 run-outs", data.loc[~(low & runout)]),
        ("Omit 75 run-outs", data.loc[~(transition & runout)]),
        ("Omit all run-outs", data.loc[~runout]),
        ("Only 100-150 groups", data.loc[data["delta_T_C"] >= 100.0]),
    ]


def profile_b10_interval(
    data: pd.DataFrame,
    delta_t: float = 35.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Ordinary profile-likelihood interval for B10 at one excursion."""

    full = fit_primary_mle(data)
    theta_hat = np.asarray(full["theta"], dtype=float)
    loglik_hat = float(full["loglik"])
    b10_hat = float(weibull_b10(theta_hat, delta_t))
    log_b10_hat = math.log(b10_hat)
    cutoff = stats.chi2.ppf(confidence, df=1)

    def constrained_nll(log_b10: float) -> float:
        def objective(x: np.ndarray) -> float:
            k, beta1 = x
            beta0 = (
                log_b10
                + beta1 * math.log(delta_t)
                - math.log(-math.log(0.9)) / k
            )
            return -weibull_log_likelihood((k, beta1, beta0), data)

        best = None
        for start in (theta_hat[:2], np.array([2.0, 1.0]), np.array([4.0, 3.0])):
            result = optimize.minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=[(1.0000001, 10.0), (1.0e-6, 10.0)],
                options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 20000},
            )
            if best is None or result.fun < best.fun:
                best = result
        if best is None or not np.isfinite(best.fun):
            raise RuntimeError("Constrained profile optimisation failed")
        return float(best.fun)

    nll_hat = -loglik_hat

    def root(log_b10: float) -> float:
        return 2.0 * (constrained_nll(log_b10) - nll_hat) - cutoff

    lower = math.exp(optimize.brentq(root, log_b10_hat - 2.0, log_b10_hat))
    upper = math.exp(optimize.brentq(root, log_b10_hat, log_b10_hat + 2.0))
    return {"mle": b10_hat, "lower": lower, "upper": upper}


def complete_weibull_log_likelihood(t: np.ndarray, k: float, eta: float) -> float:
    """Log-likelihood for complete Weibull failure times."""

    return float(
        np.sum(
            np.log(k)
            - np.log(eta)
            + (k - 1.0) * (np.log(t) - np.log(eta))
            - (t / eta) ** k
        )
    )


def common_shape_lrt(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Common-shape likelihood-ratio diagnostic for the fully failed groups."""

    complete = data.loc[data["delta_T_C"] >= 100.0]
    rows: list[dict[str, float]] = []
    separate_loglik = 0.0

    for excursion, group in complete.groupby("delta_T_C", sort=True):
        times = group["cycles"].to_numpy(dtype=float)
        k_hat, _, eta_hat = stats.weibull_min.fit(times, floc=0.0)
        loglik = complete_weibull_log_likelihood(times, k_hat, eta_hat)
        separate_loglik += loglik
        rows.append(
            {
                "delta_T": float(excursion),
                "k": float(k_hat),
                "eta_cycles": float(eta_hat),
            }
        )

    def negative_common_shape_loglik(k: float) -> float:
        if k <= 0.0:
            return np.inf
        total = 0.0
        for _, group in complete.groupby("delta_T_C", sort=True):
            times = group["cycles"].to_numpy(dtype=float)
            eta = float(np.mean(times**k) ** (1.0 / k))
            total += complete_weibull_log_likelihood(times, k, eta)
        return -total

    result = optimize.minimize_scalar(
        negative_common_shape_loglik,
        bounds=(0.1, 10.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    common_loglik = float(-result.fun)
    statistic = 2.0 * (separate_loglik - common_loglik)
    p_value = float(stats.chi2.sf(statistic, df=2))

    test = {
        "loglik_common_k": common_loglik,
        "loglik_pergroup_k": float(separate_loglik),
        "lr_stat": float(statistic),
        "df": 2.0,
        "p_value": p_value,
        "common_k_hat": float(result.x),
    }
    return pd.DataFrame(rows), test


def classical_rhat(chains: np.ndarray) -> np.ndarray:
    """Classical Gelman-Rubin R-hat for an array of shape (chains, draws, parameters)."""

    chains = np.asarray(chains, dtype=float)
    m, n, p = chains.shape
    output = np.full(p, np.nan)
    if m < 2 or n < 2:
        return output

    for j in range(p):
        x = chains[:, :, j]
        chain_means = np.mean(x, axis=1)
        chain_variances = np.var(x, axis=1, ddof=1)
        within = float(np.mean(chain_variances))
        between = float(n * np.var(chain_means, ddof=1))
        if within <= 0.0:
            continue
        variance = ((n - 1.0) / n) * within + between / n
        output[j] = math.sqrt(max(variance / within, 0.0))
    return output


def _autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    if np.allclose(values, 0.0):
        return np.ones(values.size)

    n = values.size
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    spectrum = np.fft.rfft(values, n=n_fft)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[:n]
    covariance /= np.arange(n, 0, -1)
    return covariance / covariance[0]


def effective_sample_size(chains: np.ndarray) -> float:
    """Initial-positive-sequence effective sample size for one parameter."""

    values = np.asarray(chains, dtype=float)
    m, n = values.shape
    if n < 4:
        return float(m * n)

    autocorrelation = np.mean(
        np.vstack([_autocorrelation_fft(values[i]) for i in range(m)]),
        axis=0,
    )
    positive_sum = 0.0
    lag = 1
    while lag + 1 < n:
        pair = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair < 0.0:
            break
        positive_sum += pair
        lag += 2

    tau = 1.0 + 2.0 * positive_sum
    if tau <= 0.0:
        return float(m * n)
    return float(np.clip((m * n) / tau, 1.0, m * n))



def curved_weibull_log_likelihood(theta: Iterable[float], data: pd.DataFrame) -> float:
    """Right-censored Weibull log-likelihood with a quadratic log-stress term."""

    k, beta1, beta0, beta2 = map(float, theta)
    if k <= 1.0 or beta1 <= 0.0:
        return -np.inf

    t = data["cycles"].to_numpy(dtype=float)
    x = np.log(data["delta_T_C"].to_numpy(dtype=float))
    event = data["event"].to_numpy(dtype=float)
    xbar = float(np.mean(np.log(np.sort(data["delta_T_C"].unique()))))
    log_scale = beta0 - beta1 * x + beta2 * (x - xbar) ** 2
    log_ratio = np.log(t) - log_scale
    cumulative_hazard = np.exp(np.clip(k * log_ratio, -700.0, 700.0))
    contribution = (
        event * (np.log(k) + (k - 1.0) * log_ratio - log_scale)
        - cumulative_hazard
    )
    return float(np.sum(contribution))


def fit_curved_weibull(data: pd.DataFrame, beta2_sd: float = 0.5) -> dict[str, float | list[float]]:
    """Fit the curvature sensitivity model with Gaussian shrinkage on beta2."""

    bounds = [(1.001, 25.0), (1.0e-4, 8.0), (-5.0, 30.0), (-2.0, 2.0)]
    starts = (
        (3.0, 2.4, 17.6, 0.0),
        (3.0, 2.0, 16.0, -0.3),
        (4.0, 3.0, 20.0, 0.3),
        (2.0, 1.0, 12.0, -0.5),
    )

    def objective(theta: np.ndarray) -> float:
        loglik = curved_weibull_log_likelihood(theta, data)
        if not np.isfinite(loglik):
            return np.inf
        penalty = 0.5 * (float(theta[3]) / beta2_sd) ** 2
        return -loglik + penalty

    best = None
    for start in starts:
        result = optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 20000},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Curved Weibull optimisation failed")

    theta = np.asarray(best.x, dtype=float)
    loglik = curved_weibull_log_likelihood(theta, data)
    n = len(data)
    return {
        "theta": theta.tolist(),
        "loglik": float(loglik),
        "aic": float(-2.0 * loglik + 2.0 * 4),
        "bic": float(-2.0 * loglik + 4.0 * np.log(n)),
        "beta2_penalty_sd": float(beta2_sd),
        "converged": bool(best.success),
    }


def lognormal_log_likelihood(theta: Iterable[float], data: pd.DataFrame) -> float:
    """Exact right-censored lognormal AFT log-likelihood."""

    sigma, beta1, beta0 = map(float, theta)
    if sigma <= 0.0 or beta1 <= 0.0:
        return -np.inf

    t = data["cycles"].to_numpy(dtype=float)
    x = np.log(data["delta_T_C"].to_numpy(dtype=float))
    event = data["event"].to_numpy(dtype=float)
    mu = beta0 - beta1 * x
    z = (np.log(t) - mu) / sigma
    failure_log_density = (
        -np.log(t * sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * z**2
    )
    survival_log_probability = stats.norm.logsf(z)
    return float(
        np.sum(event * failure_log_density + (1.0 - event) * survival_log_probability)
    )


def fit_lognormal_aft(data: pd.DataFrame) -> dict[str, float | list[float]]:
    """Fit the censored lognormal accelerated-failure-time comparator."""

    bounds = [(1.0e-3, 5.0), (1.0e-4, 8.0), (-5.0, 30.0)]
    starts = ((0.4, 2.2, 16.3), (0.5, 2.0, 15.0), (1.0, 3.0, 20.0))
    best = None
    for start in starts:
        result = optimize.minimize(
            lambda x: -lognormal_log_likelihood(x, data),
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 20000},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Lognormal AFT optimisation failed")

    theta = np.asarray(best.x, dtype=float)
    loglik = lognormal_log_likelihood(theta, data)
    n = len(data)
    return {
        "theta": theta.tolist(),
        "loglik": float(loglik),
        "aic": float(-2.0 * loglik + 2.0 * 3),
        "bic": float(-2.0 * loglik + 3.0 * np.log(n)),
        "converged": bool(best.success),
    }


def waic_dic(samples: np.ndarray, data: pd.DataFrame) -> dict[str, float]:
    """Compute WAIC and DIC for retained primary-model posterior draws."""

    draws = np.asarray(samples, dtype=float).reshape(-1, 3)
    t = data["cycles"].to_numpy(dtype=float)
    d = data["delta_T_C"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=float)

    loglik_matrix = np.empty((draws.shape[0], len(data)), dtype=float)
    for index, (k, beta1, beta0) in enumerate(draws):
        log_scale = beta0 - beta1 * np.log(d)
        log_ratio = np.log(t) - log_scale
        loglik_matrix[index] = (
            event * (np.log(k) + (k - 1.0) * log_ratio - log_scale)
            - np.exp(np.clip(k * log_ratio, -700.0, 700.0))
        )

    max_loglik = np.max(loglik_matrix, axis=0)
    lppd = np.sum(
        max_loglik
        + np.log(np.mean(np.exp(loglik_matrix - max_loglik), axis=0))
    )
    p_waic = float(np.sum(np.var(loglik_matrix, axis=0, ddof=1)))
    waic = float(-2.0 * (lppd - p_waic))

    deviance = -2.0 * np.sum(loglik_matrix, axis=1)
    d_bar = float(np.mean(deviance))
    theta_bar = np.mean(draws, axis=0)
    d_hat = float(-2.0 * weibull_log_likelihood(theta_bar, data))
    p_dic = d_bar - d_hat
    dic = d_hat + 2.0 * p_dic

    return {
        "waic": waic,
        "p_waic": p_waic,
        "dic": dic,
        "p_dic": p_dic,
    }

def load_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    """Write a deterministic, human-readable JSON file."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
