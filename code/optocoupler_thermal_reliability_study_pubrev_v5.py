
#!/usr/bin/env python3
"""
Optocoupler thermal-cycling reliability study v5.0 (single-script deliverable).

Implements:
1) Unit-level dataset assembly with right-censoring
2) Primary Weibull-Coffin-Manson model (PICM-L): MLE, Firth-penalized MLE
3) Optional curvature sensitivity Weibull model (PICM-C)
4) Adaptive random-walk Metropolis Bayesian inference for PICM-L (3 chains)
5) WAIC (primary Bayesian comparison) and DIC (supplementary)
6) Lognormal AFT sensitivity model with exact censoring
7) Uncorrected and Bartlett-corrected profile likelihood CI for reference-case B10
8) Posterior predictive reliability quantities
9) Publication-style tables and figures
10) Additional diagnostics, validation checks, and submission-ready outputs
11) Publication-readability figure repair, QA report, and GitHub package creation

Dependencies: numpy, scipy, pandas, matplotlib
"""

import math
import json
import shutil
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy import optimize, special, stats


# =============================================================================
# User-settable run controls
# =============================================================================

DELTA_T_REFERENCE = 35.0  # near-boundary reference thermal swing in deg C
DELTA_T_USE = DELTA_T_REFERENCE  # reference-case alias used across calculations
CYCLE_TIME_HOURS = 3.0
INPUT_CSV = Path("optocoupler_ttf_unit_level.csv")

# Low-deltaT extrapolation targets and decision metrics
TARGET_DELTA_TS = np.array([10.0, 15.0, 20.0, 25.0, 35.0], dtype=float)
MISSION_CYCLE_TARGETS = [5000.0, 10000.0]
B10_GRID_DELTA_T = np.arange(10.0, 151.0, 1.0)
PRIOR_SENS_DELTA_TARGETS = np.array([20.0, 25.0, 35.0], dtype=float)
PRIOR_VS_DATA_TARGETS = np.array([20.0, 25.0, 35.0], dtype=float)
HIGH_DT_TARGETS = np.array([175.0, 200.0], dtype=float)
HIGH_DT_GRID_DELTA_T = np.arange(100.0, 200.0 + 1.0, 1.0, dtype=float)
LOW_STRESS_DURATION_TARGETS = np.array([20.0, 25.0], dtype=float)
DECISION_PROB_DELTA_GRID = np.arange(10.0, 50.0 + 0.25, 0.25, dtype=float)
DECISION_PROBABILITY_LEVELS = np.array([0.50, 0.90, 0.95], dtype=float)
OBSERVED_DOMAIN_MIN = 25.0
OBSERVED_DOMAIN_MAX = 150.0
EXTRAPOLATION_CUTOFF = 25.0
LOW_INFO_TESTED_MAX = 50.0
LOWEST_FAILURE_OBSERVED_STRESS = 75.0
REFERENCE_MARKERS = np.array([10.0, 15.0, 25.0, 35.0], dtype=float)

# Optional strict structure for the real uploaded experiment.
ENFORCE_REAL_FAILURE_PATTERN = True
REAL_PATTERN_MISMATCH_IS_ERROR = False

OUT_DIR = Path("results_optocoupler_reliability_v5")
FIG_DIR = OUT_DIR / "figures"
ALL_FIG_DIR = FIG_DIR / "all"
PAPER_FIG_DIR = FIG_DIR / "paper"
MAIN_FIG_DIR = FIG_DIR / "_main_logical"
SUPP_FIG_DIR = FIG_DIR / "_supplementary_logical"
TAB_DIR = OUT_DIR / "tables"

N_OPT_STARTS_MAIN = 40
N_OPT_STARTS_BOOT = 8

MCMC_BURNIN = 12000
MCMC_KEEP = 20000
MCMC_ADAPT_BLOCK = 100

# Draw count for sensitivity-model uncertainty via Laplace approximation.
N_SENSITIVITY_DRAWS = 20000

N_PROFILE_GRID = 61
N_BARTLETT_BOOT = 120

SEED_OPT = 20260301
SEED_MCMC = [20260311, 20260312, 20260313]
SEED_BOOT = 20260321
SEED_PRED = 20260331

# Prior sensitivity run controls (kept shorter than the primary run for tractability).
PRIOR_SENS_BURNIN = 4000
PRIOR_SENS_KEEP = 6000
PRIOR_SENS_DRAW_SUBSAMPLE = 12000
N_PRIOR_ONLY_DRAWS = 60000
N_GROUP_SHAPE_BOOT = 2000
MCMC_TARGET_ACCEPT = 0.234
MCMC_ACCEPT_LOWER = 0.20
MCMC_ACCEPT_UPPER = 0.40

LOSO_MCMC_BURNIN = 2500
LOSO_MCMC_KEEP = 3000
LOSO_MCMC_ADAPT_BLOCK = 100
LOSO_LAPLACE_DRAWS = 5000

# Prior settings (centralized and explicit).
PRIOR_SETTINGS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "k_gamma_shape": 9.0,
        "k_gamma_rate": 3.0,
        "k_trunc_lower": 1.0,
        "beta1_mean": 2.5,
        "beta1_sd": 0.8,
        "beta1_trunc_lower": 0.5,
        "beta0_mean": 18.0,
        "beta0_sd": 4.0,
        "beta2_sd": 0.5,
    },
    "diffuse": {
        "k_gamma_shape": 6.0,
        "k_gamma_rate": 2.0,
        "k_trunc_lower": 1.0,
        "beta1_mean": 2.5,
        "beta1_sd": 1.6,
        "beta1_trunc_lower": 0.5,
        "beta0_mean": 18.0,
        "beta0_sd": 8.0,
        "beta2_sd": 0.9,
    },
    "conservative": {
        "k_gamma_shape": 12.0,
        "k_gamma_rate": 4.0,
        "k_trunc_lower": 1.0,
        "beta1_mean": 2.5,
        "beta1_sd": 0.5,
        "beta1_trunc_lower": 0.5,
        "beta0_mean": 18.0,
        "beta0_sd": 2.5,
        "beta2_sd": 0.3,
    },
}

BASELINE_PRIOR_KEY = "baseline"
PRIOR_SENSITIVITY_KEYS = ["baseline", "diffuse", "conservative"]

# Plotting / validation switches
ENABLE_LEAVE_ONE_STRESS_OUT = True
LOSO_MAX_STRESS = None  # Set int to limit number of held-out stresses for speed.

EXPECTED_STRESS_LEVELS = np.array([25.0, 50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)
EXPECTED_REAL_FAILURE_PATTERN = {
    25.0: (0, 8),
    50.0: (0, 8),
    75.0: (5, 3),
    100.0: (8, 0),
    125.0: (8, 0),
    150.0: (8, 0),
}

PLOT_COLORS = {
    "bayes_fit": "#355c9a",
    "bayes_band": "#d6e4f5",
    "km_empirical": "#8f4b3e",
    "posterior_density": "#2e7d73",
    "mle": "#3f4752",
    "firth": "#8b6b47",
    "picm_c": "#a33d34",
    "failed": "#4a4a4a",
    "censored_edge": "#4a4a4a",
    "aux": "#2f3742",
    "chain_2": "#2e7d73",
}

IEEE_ONE_COL = 3.5
IEEE_ONE_COL_WIDE = 3.75
IEEE_TWO_COL = 7.16
IEEE_TWO_COL_WIDE = IEEE_TWO_COL
MIN_PANEL_TICK_PT = 7.0

PUB_FIGURE_MANIFEST: List[Dict[str, object]] = []

SLOT_MAP: Dict[str, Tuple[str, str]] = {
    "figure_b10_vs_deltaT_regime": (
        "Fig. 1",
        "B10 versus protocol-defined DeltaT with model uncertainty and evidential regimes.",
    ),
    "figure_failure_fraction_calibration": (
        "Fig. 2",
        "Observed and posterior-predictive failure fractions across tested stress levels.",
    ),
    "figure_stress_life_regime": (
        "Fig. 3",
        "Stress-life data with censoring, fitted model summaries and evidential regimes.",
    ),
    "main_reference_case_b10_comparison": (
        "Fig. 4",
        "Reference-case B10 estimates and interval summaries.",
    ),
    "fig_group_shape_sensitivity": (
        "Fig. 5",
        "Group-specific Weibull shape sensitivity and common-shape likelihood-ratio test.",
    ),
    "fig_model_conditional_decision_boundary_uncertainty": (
        "Fig. 6",
        "Model-conditional posterior probability that B10 exceeds mission-cycle targets.",
    ),
    "fig_prior_vs_data_b10": (
        "Fig. S1",
        "Prior-only and data-informed B10 projections at low and near-boundary stresses.",
    ),
    "figure_b10_density_10C_15C": (
        "Fig. S2",
        "Model-form sensitivity of extrapolative B10 distributions at 10 C and 15 C.",
    ),
    "main_survival_panels_pub": (
        "Fig. S3",
        "Stress-wise survival curves with Kaplan-Meier summaries and posterior intervals.",
    ),
    "supp_leave_one_stress_out": (
        "Fig. S4",
        "Leave-one-stress-out observed versus predicted failure fractions.",
    ),
    "supp_prior_sensitivity_b10_35C": (
        "Fig. S5",
        "Prior-sensitivity B10 intervals at 20, 25, and 35 C.",
    ),
    "supp_prior_vs_posterior_refined": (
        "Fig. S6",
        "Baseline prior versus data-informed posterior interval summaries.",
    ),
    "supp_profile_likelihood_b10_refined": (
        "Fig. S7",
        "Profile-likelihood diagnostics for reference-case B10.",
    ),
    "supp_trace_plots_refined": (
        "Fig. S8",
        "MCMC trace diagnostics for retained posterior samples.",
    ),
    "supp_weibull_probability_plots_refined": (
        "Fig. S9",
        "Stress-wise Weibull probability diagnostics.",
    ),
    "supp_weibull_residual_diagnostics": (
        "Fig. S10",
        "Cox-Snell and residual quantile diagnostics.",
    ),
    "figure_high_deltaT_extrapolation_CI": (
        "Fig. S11",
        "High-side model-conditional B10 extrapolation with posterior uncertainty.",
    ),
}


# =============================================================================
# Data container
# =============================================================================


@dataclass
class ReliabilityData:
    """Convenience container for reliability dataset arrays."""

    df: pd.DataFrame
    x_log: np.ndarray
    y: np.ndarray
    event: np.ndarray
    stress_levels: np.ndarray
    xbar: float
    n: int
    stress_to_idx: Dict[float, np.ndarray]


# =============================================================================
# Dataset loading and validation
# =============================================================================


def resolve_input_csv(input_csv: Path) -> Path:
    """Resolve input CSV in workspace-root and packaged single-script layouts."""
    if input_csv.is_absolute():
        return input_csv
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / input_csv,
        script_dir / input_csv,
        script_dir.parent / input_csv,
        script_dir.parent / "data" / input_csv.name,
        Path.cwd() / "data" / input_csv.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_input_dataset(input_csv: Path, cycle_time_hours: float) -> Tuple[pd.DataFrame, str, Path]:
    """Load the unit-level time-to-failure dataset from the packaged CSV."""
    csv_path = resolve_input_csv(input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}. "
            "Set INPUT_CSV to the unit-level dataset or place the CSV under the package data folder."
        )
    raw = pd.read_csv(csv_path)

    # Accept either the canonical schema or the uploaded experimental schema.
    canonical_cols = {"unit_id", "delta_T", "ttf_hours", "cycles", "event"}
    uploaded_cols = {"experiment_number", "delta_T_C", "sample_number", "ttf_hours", "cycles_to_failure_approx", "event"}

    if canonical_cols.issubset(set(raw.columns)):
        df = raw.copy()
    elif uploaded_cols.issubset(set(raw.columns)):
        df = pd.DataFrame()
        df["unit_id"] = raw.apply(
            lambda r: f"E{int(r['experiment_number']):02d}_U{int(r['sample_number']):02d}",
            axis=1,
        )
        df["delta_T"] = raw["delta_T_C"]
        df["ttf_hours"] = raw["ttf_hours"]
        # Enforce physical definition exactly for analysis.
        df["cycles"] = raw["ttf_hours"] / cycle_time_hours
        df["event"] = raw["event"]
        if "cycles_to_failure_approx" in raw.columns:
            diff = np.max(np.abs((raw["ttf_hours"] / cycle_time_hours) - raw["cycles_to_failure_approx"]))
            if float(diff) > 1e-9:
                warnings.warn(
                    "Uploaded 'cycles_to_failure_approx' differs from exact ttf_hours/3; "
                    "analysis uses exact cycles = ttf_hours/3."
                )
    else:
        raise ValueError(
            "Input CSV columns not recognized. Expected either canonical columns "
            "['unit_id','delta_T','ttf_hours','cycles','event'] or uploaded schema "
            "['experiment_number','delta_T_C','sample_number','ttf_hours','cycles_to_failure_approx','event',...]."
        )

    return df, "csv_file", csv_path


def validate_dataset(
    df: pd.DataFrame,
    cycle_time_hours: float = 3.0,
    enforce_real_failure_pattern: bool = True,
    pattern_mismatch_is_error: bool = False,
) -> Dict[str, object]:
    """Validate dataset structure and constraints and optionally real failure/censor pattern."""
    required_cols = ["unit_id", "delta_T", "ttf_hours", "cycles", "event"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Enforce consistent dtypes for downstream processing.
    df["unit_id"] = df["unit_id"].astype(str)
    for col in ["delta_T", "ttf_hours", "cycles"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["event"] = pd.to_numeric(df["event"], errors="raise").astype(int)

    if df.shape[0] != 48:
        raise ValueError(f"Expected 48 rows (6 stresses x 8 units), got {df.shape[0]}")

    stress_found = np.array(sorted(df["delta_T"].unique()), dtype=float)
    if not np.array_equal(EXPECTED_STRESS_LEVELS, stress_found):
        raise ValueError(f"Stress levels mismatch. Expected {EXPECTED_STRESS_LEVELS}, found {stress_found}")

    counts = df.groupby("delta_T")["unit_id"].count()
    if not np.all(counts.values == 8):
        raise ValueError("Each stress level must have exactly 8 units.")

    if (df["delta_T"] <= 0).any():
        raise ValueError("All delta_T values must be > 0.")
    if (df["ttf_hours"] <= 0).any() or (df["cycles"] <= 0).any():
        raise ValueError("All ttf_hours and cycles must be > 0.")
    if not set(df["event"].unique()).issubset({0, 1}):
        raise ValueError("event must be 0 or 1.")

    ratio = df["ttf_hours"].values / cycle_time_hours
    if not np.allclose(ratio, df["cycles"].values, rtol=0.0, atol=1e-12):
        raise ValueError("cycles must equal ttf_hours / cycle_time_hours exactly.")

    failure_summary = (
        df.groupby("delta_T", as_index=False)
        .agg(n_failed=("event", lambda s: int(np.sum(s == 1))), n_censored=("event", lambda s: int(np.sum(s == 0))))
        .sort_values("delta_T")
        .reset_index(drop=True)
    )
    failure_summary["delta_T"] = failure_summary["delta_T"].astype(int)
    failure_summary["n_failed"] = failure_summary["n_failed"].astype(int)
    failure_summary["n_censored"] = failure_summary["n_censored"].astype(int)

    strict_passed = True
    strict_mismatches = []
    if enforce_real_failure_pattern:
        for s in EXPECTED_STRESS_LEVELS:
            got = failure_summary.loc[failure_summary["delta_T"] == int(s), ["n_failed", "n_censored"]].iloc[0]
            expected_f, expected_c = EXPECTED_REAL_FAILURE_PATTERN[float(s)]
            if (int(got["n_failed"]) != expected_f) or (int(got["n_censored"]) != expected_c):
                strict_passed = False
                strict_mismatches.append(
                    f"DeltaT={int(s)}C expected (failed={expected_f}, censored={expected_c}) "
                    f"but got (failed={int(got['n_failed'])}, censored={int(got['n_censored'])})"
                )

        if not strict_passed:
            msg = "Real-pattern validation mismatch:\n  " + "\n  ".join(strict_mismatches)
            if pattern_mismatch_is_error:
                raise ValueError(msg)
            warnings.warn(msg)

    return {
        "stress_found": stress_found,
        "failure_summary": failure_summary,
        "strict_pattern_passed": strict_passed if enforce_real_failure_pattern else None,
        "strict_pattern_mismatches": strict_mismatches,
    }


def print_data_provenance(
    source_kind: str,
    csv_path: Path,
    df: pd.DataFrame,
    validation_info: Dict[str, object],
) -> None:
    """Print run provenance summary to avoid accidental wrong data source usage."""
    print("\n================= Data Provenance =================")
    print("Data source: CSV file")
    print(f"Input file: {csv_path.resolve()}")
    print(f"Rows: {df.shape[0]}")
    stress_levels = [int(v) for v in validation_info["stress_found"]]
    print(f"Stress levels found: {stress_levels}")
    print("Failure/censor counts by stress:")
    for _, r in validation_info["failure_summary"].iterrows():
        print(f"  DeltaT={int(r['delta_T'])}C -> failed={int(r['n_failed'])}, censored={int(r['n_censored'])}")
    strict_state = validation_info.get("strict_pattern_passed")
    if strict_state is None:
        print("Real-data strict validation: not enforced")
    else:
        print(f"Real-data strict validation passed: {bool(strict_state)}")
    print("===================================================\n")


def prepare_data(df: pd.DataFrame) -> ReliabilityData:
    """Prepare vectorized arrays and indexing helpers."""
    x_log = np.log(df["delta_T"].to_numpy(dtype=float))
    y = df["cycles"].to_numpy(dtype=float)
    event = df["event"].to_numpy(dtype=int)
    stress_levels = np.array(sorted(df["delta_T"].unique()), dtype=float)
    xbar = float(np.mean(np.log(stress_levels)))

    stress_to_idx: Dict[float, np.ndarray] = {}
    for s in stress_levels:
        stress_to_idx[float(s)] = np.where(df["delta_T"].to_numpy(dtype=float) == s)[0]

    return ReliabilityData(
        df=df.copy(),
        x_log=x_log,
        y=y,
        event=event,
        stress_levels=stress_levels,
        xbar=xbar,
        n=df.shape[0],
        stress_to_idx=stress_to_idx,
    )


def write_canonical_unit_level_csv(data: ReliabilityData, out_file: Path) -> pd.DataFrame:
    """Write the analysis dataset with cycles recomputed from ttf_hours/3.0."""
    required = ["unit_id", "delta_T", "ttf_hours", "event"]
    missing = [c for c in required if c not in data.df.columns]
    if missing:
        raise ValueError(f"Canonical CSV cannot be written; missing columns: {missing}")
    canonical = data.df[required].copy()
    canonical["cycles"] = canonical["ttf_hours"].astype(float) / 3.0
    canonical = canonical[["unit_id", "delta_T", "ttf_hours", "cycles", "event"]]
    expected = canonical["ttf_hours"].to_numpy(dtype=float) / 3.0
    if not np.array_equal(canonical["cycles"].to_numpy(dtype=float), expected):
        raise AssertionError("Canonical in-memory cycles are not exactly ttf_hours/3.0.")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_file, index=False, float_format="%.17g")
    round_trip = pd.read_csv(out_file)
    if not np.allclose(
        round_trip["cycles"].to_numpy(dtype=float),
        round_trip["ttf_hours"].to_numpy(dtype=float) / 3.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Canonical CSV round-trip cycles do not match ttf_hours/3.0.")
    return canonical


def make_table_a(data: ReliabilityData) -> pd.DataFrame:
    """Table A: stress-wise failure/censoring and cycle summary."""
    out = (
        data.df.groupby("delta_T", as_index=False)
        .agg(
            n_failed=("event", lambda s: int(np.sum(s == 1))),
            n_censored=("event", lambda s: int(np.sum(s == 0))),
            min_cycles=("cycles", "min"),
            max_cycles=("cycles", "max"),
            median_cycles=("cycles", "median"),
        )
        .sort_values("delta_T")
        .reset_index(drop=True)
    )
    out["delta_T"] = out["delta_T"].astype(int)
    out["n_failed"] = out["n_failed"].astype(int)
    out["n_censored"] = out["n_censored"].astype(int)
    return out


def build_censoring_summary_by_stress(data: ReliabilityData) -> pd.DataFrame:
    """Stress-wise censoring burden summary for reporting."""
    rows = []
    for s in sorted(data.stress_levels):
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        ev = data.event[idx].astype(int)
        n_total = int(len(idx))
        n_failed = int(np.sum(ev == 1))
        n_censored = int(np.sum(ev == 0))
        rows.append(
            {
                "delta_T": float(s),
                "sample_size": n_total,
                "failed_count": n_failed,
                "censored_count": n_censored,
                "censoring_fraction": float(n_censored / max(n_total, 1)),
                "max_observed_cycles": float(np.max(y)),
                "fully_censored_group": bool(n_failed == 0),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_T").reset_index(drop=True)


# =============================================================================
# Likelihoods and priors
# =============================================================================


def get_prior_config(prior_key: str = BASELINE_PRIOR_KEY) -> Dict[str, float]:
    """Return validated prior configuration for a named setting."""
    if prior_key not in PRIOR_SETTINGS:
        raise KeyError(f"Unknown prior setting '{prior_key}'. Available: {list(PRIOR_SETTINGS.keys())}")
    cfg = dict(PRIOR_SETTINGS[prior_key])
    required = [
        "k_gamma_shape",
        "k_gamma_rate",
        "k_trunc_lower",
        "beta1_mean",
        "beta1_sd",
        "beta1_trunc_lower",
        "beta0_mean",
        "beta0_sd",
        "beta2_sd",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Prior setting '{prior_key}' is missing keys: {missing}")
    if cfg["k_gamma_shape"] <= 0 or cfg["k_gamma_rate"] <= 0:
        raise ValueError(f"Prior setting '{prior_key}' has non-positive gamma hyperparameters for k.")
    if cfg["beta1_sd"] <= 0 or cfg["beta0_sd"] <= 0 or cfg["beta2_sd"] <= 0:
        raise ValueError(f"Prior setting '{prior_key}' has non-positive prior scale(s).")
    return cfg


def truncated_gamma_logpdf(x: float, shape: float, rate: float, lower: float) -> float:
    """
    Log-density for Gamma(shape, rate) truncated to x > lower.

    Effective prior is:
      p(x | x>lower) = p_gamma(x) / P_gamma(X>lower), x>lower.
    """
    xv = float(x)
    if xv <= lower:
        return -np.inf
    scale = 1.0 / float(rate)
    sf_lower = float(stats.gamma.sf(lower, a=shape, scale=scale))
    if sf_lower <= 0.0 or (not np.isfinite(sf_lower)):
        return -np.inf
    return float(stats.gamma.logpdf(xv, a=shape, scale=scale) - np.log(sf_lower))


def summarize_truncated_gamma_prior(
    shape: float,
    rate: float,
    lower: float,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> Dict[str, float]:
    """Numerically summarize a lower-truncated Gamma prior."""
    shp = float(shape)
    rte = float(rate)
    low = float(lower)
    if (shp <= 0.0) or (rte <= 0.0):
        raise ValueError("shape and rate must be positive.")
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("All quantiles must be in (0,1).")

    scale = 1.0 / rte
    sf_low = float(stats.gamma.sf(low, a=shp, scale=scale))
    cdf_low = float(stats.gamma.cdf(low, a=shp, scale=scale))
    if (sf_low <= 0.0) or (not np.isfinite(sf_low)):
        raise RuntimeError("Truncated Gamma normalizing probability is invalid.")

    upper_reg_a = float(special.gammaincc(shp, rte * low))
    upper_reg_a1 = float(special.gammaincc(shp + 1.0, rte * low))
    upper_reg_a2 = float(special.gammaincc(shp + 2.0, rte * low))
    mean_eff = (shp / rte) * (upper_reg_a1 / upper_reg_a)
    second_eff = (shp * (shp + 1.0) / (rte**2)) * (upper_reg_a2 / upper_reg_a)
    var_eff = max(second_eff - mean_eff**2, 0.0)

    out = {
        "shape": shp,
        "rate": rte,
        "lower_truncation": low,
        "normalizing_prob_sf_lower": sf_low,
        "mean_effective": float(mean_eff),
        "sd_effective": float(np.sqrt(var_eff)),
    }
    for q in quantiles:
        target_cdf = cdf_low + q * sf_low
        q_label = f"{100.0 * q:.1f}".replace(".", "_")
        out[f"q{q_label}_effective"] = float(stats.gamma.ppf(target_cdf, a=shp, scale=scale))
    return out


def build_prior_hyperparameter_table() -> pd.DataFrame:
    """Build machine-readable table of prior hyperparameters used by the code."""
    rows = []
    for key in PRIOR_SETTINGS:
        cfg = get_prior_config(key)
        rows.append(
            {
                "prior_setting": key,
                "beta0_prior_mean": cfg["beta0_mean"],
                "beta0_prior_sd": cfg["beta0_sd"],
                "beta1_prior_mean": cfg["beta1_mean"],
                "beta1_prior_sd": cfg["beta1_sd"],
                "beta1_trunc_lower": cfg["beta1_trunc_lower"],
                "k_gamma_shape": cfg["k_gamma_shape"],
                "k_gamma_rate": cfg["k_gamma_rate"],
                "k_trunc_lower": cfg["k_trunc_lower"],
                "beta2_prior_sd": cfg["beta2_sd"],
                "k_prior_note": "Gamma(shape,rate) truncated to k>k_trunc_lower",
            }
        )
    return pd.DataFrame(rows)


def build_effective_k_prior_summary_table() -> pd.DataFrame:
    """Build effective summary table for the truncated-Gamma prior on k."""
    rows = []
    for key in PRIOR_SETTINGS:
        cfg = get_prior_config(key)
        s = summarize_truncated_gamma_prior(
            shape=cfg["k_gamma_shape"],
            rate=cfg["k_gamma_rate"],
            lower=cfg["k_trunc_lower"],
        )
        rows.append(
            {
                "prior_setting": key,
                "k_prior_family": "Gamma_truncated_lower",
                "k_trunc_lower": s["lower_truncation"],
                "k_effective_mean": s["mean_effective"],
                "k_effective_sd": s["sd_effective"],
                "k_effective_q2_5": s["q2_5_effective"],
                "k_effective_q50": s["q50_0_effective"],
                "k_effective_q97_5": s["q97_5_effective"],
                "normalizing_prob_k_gt_lower": s["normalizing_prob_sf_lower"],
            }
        )
    return pd.DataFrame(rows)


def _safe_exp(x: np.ndarray, cap: float = 700.0) -> np.ndarray:
    """Exponentiation with clipping to avoid overflow."""
    return np.exp(np.clip(x, -745.0, cap))


def loglik_weibull_picm_l(theta: np.ndarray, data: ReliabilityData, pointwise: bool = False):
    """
    Exact right-censored Weibull PICM-L log-likelihood.
    theta = [k, beta1, beta0]
    log(eta_i) = beta0 - beta1 * log(delta_T_i)
    """
    k, beta1, beta0 = [float(v) for v in theta]
    if (k <= 1.0) or (beta1 <= 0.0) or (not np.isfinite(beta0)):
        return np.full(data.n, -np.inf) if pointwise else -np.inf

    log_eta = beta0 - beta1 * data.x_log
    if not np.all(np.isfinite(log_eta)):
        return np.full(data.n, -np.inf) if pointwise else -np.inf

    log_ratio = np.log(data.y) - log_eta  # log(y/eta)
    expo = k * log_ratio
    if np.any(expo > 700.0):
        return np.full(data.n, -np.inf) if pointwise else -np.inf
    z = _safe_exp(expo)

    ll_i = data.event * (np.log(k) + (k - 1.0) * log_ratio - log_eta) - z
    if not np.all(np.isfinite(ll_i)):
        return np.full(data.n, -np.inf) if pointwise else -np.inf
    if pointwise:
        return ll_i
    return float(np.sum(ll_i))


def nll_weibull_picm_l(theta: np.ndarray, data: ReliabilityData) -> float:
    """Negative log-likelihood for PICM-L."""
    ll = loglik_weibull_picm_l(theta, data, pointwise=False)
    if not np.isfinite(ll):
        return np.inf
    return -ll


def loglik_weibull_picm_c(theta: np.ndarray, data: ReliabilityData, pointwise: bool = False):
    """
    Exact right-censored Weibull PICM-C log-likelihood.
    theta = [k, beta1, beta0, beta2]
    log(eta_i) = beta0 - beta1*log(delta_T_i) + beta2*(log(delta_T_i)-xbar)^2
    """
    k, beta1, beta0, beta2 = [float(v) for v in theta]
    if (k <= 1.0) or (beta1 <= 0.0):
        return np.full(data.n, -np.inf) if pointwise else -np.inf

    log_eta = beta0 - beta1 * data.x_log + beta2 * (data.x_log - data.xbar) ** 2
    log_ratio = np.log(data.y) - log_eta
    expo = k * log_ratio
    if np.any(expo > 700.0):
        return np.full(data.n, -np.inf) if pointwise else -np.inf
    z = _safe_exp(expo)
    ll_i = data.event * (np.log(k) + (k - 1.0) * log_ratio - log_eta) - z
    if not np.all(np.isfinite(ll_i)):
        return np.full(data.n, -np.inf) if pointwise else -np.inf
    if pointwise:
        return ll_i
    return float(np.sum(ll_i))


def nll_weibull_picm_c(theta: np.ndarray, data: ReliabilityData) -> float:
    """Negative log-likelihood for PICM-C."""
    ll = loglik_weibull_picm_c(theta, data, pointwise=False)
    if not np.isfinite(ll):
        return np.inf
    return -ll


def loglik_lognormal_aft(theta: np.ndarray, data: ReliabilityData, pointwise: bool = False):
    """
    Exact right-censored lognormal AFT log-likelihood.
    theta = [sigma, beta1, beta0]
    log(Y_ij) ~ N(mu_i, sigma^2), mu_i = beta0 - beta1*log(delta_T_i)
    """
    sigma, beta1, beta0 = [float(v) for v in theta]
    if (sigma <= 0.0) or (beta1 <= 0.0):
        return np.full(data.n, -np.inf) if pointwise else -np.inf

    mu = beta0 - beta1 * data.x_log
    log_y = np.log(data.y)

    # For failures on Y-scale, include Jacobian: f_Y(y)=f_logY(log y)/y.
    ll_fail = stats.norm.logpdf(log_y, loc=mu, scale=sigma) - log_y
    ll_cens = stats.norm.logsf(log_y, loc=mu, scale=sigma)
    ll_i = data.event * ll_fail + (1 - data.event) * ll_cens
    if not np.all(np.isfinite(ll_i)):
        return np.full(data.n, -np.inf) if pointwise else -np.inf
    if pointwise:
        return ll_i
    return float(np.sum(ll_i))


def nll_lognormal_aft(theta: np.ndarray, data: ReliabilityData) -> float:
    """Negative log-likelihood for lognormal AFT."""
    ll = loglik_lognormal_aft(theta, data, pointwise=False)
    if not np.isfinite(ll):
        return np.inf
    return -ll


def logprior_picm_l(theta: np.ndarray, prior_cfg: Optional[Dict[str, float]] = None) -> float:
    """
    Prior for PICM-L with explicit truncation:
    - k ~ Gamma(shape, rate), truncated to k > k_trunc_lower
    - beta1 ~ Truncated Normal(mean, sd^2), lower=beta1_trunc_lower
    - beta0 ~ Normal(mean, sd^2)
    """
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    k, beta1, beta0 = [float(v) for v in theta]
    k_lower = float(cfg["k_trunc_lower"])
    b1_lower = float(cfg["beta1_trunc_lower"])
    if (k <= k_lower) or (beta1 <= b1_lower):
        return -np.inf

    a = float(cfg["k_gamma_shape"])
    rate = float(cfg["k_gamma_rate"])
    logp_k = truncated_gamma_logpdf(k, shape=a, rate=rate, lower=k_lower)
    if not np.isfinite(logp_k):
        return -np.inf

    mu_b1 = float(cfg["beta1_mean"])
    sd_b1 = float(cfg["beta1_sd"])
    a_trunc = (b1_lower - mu_b1) / sd_b1
    z_norm = 1.0 - stats.norm.cdf(a_trunc)
    if z_norm <= 0.0:
        return -np.inf
    logp_b1 = stats.norm.logpdf(beta1, loc=mu_b1, scale=sd_b1) - np.log(z_norm)

    logp_b0 = stats.norm.logpdf(beta0, loc=float(cfg["beta0_mean"]), scale=float(cfg["beta0_sd"]))
    out = logp_k + logp_b1 + logp_b0
    return float(out) if np.isfinite(out) else -np.inf


def logprior_picm_c(theta: np.ndarray, prior_cfg: Optional[Dict[str, float]] = None) -> float:
    """
    Prior for PICM-C:
    same as PICM-L for k, beta1, beta0 and beta2 ~ N(0, beta2_sd^2)
    """
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    if len(theta) != 4:
        return -np.inf
    lp_base = logprior_picm_l(theta[:3], prior_cfg=cfg)
    if not np.isfinite(lp_base):
        return -np.inf
    beta2 = float(theta[3])
    lp_b2 = stats.norm.logpdf(beta2, loc=0.0, scale=float(cfg["beta2_sd"]))
    return float(lp_base + lp_b2)


def logposterior_picm_l(theta: np.ndarray, data: ReliabilityData, prior_cfg: Optional[Dict[str, float]] = None) -> float:
    """Posterior log-density for PICM-L."""
    lp = logprior_picm_l(theta, prior_cfg=prior_cfg)
    if not np.isfinite(lp):
        return -np.inf
    ll = loglik_weibull_picm_l(theta, data, pointwise=False)
    if not np.isfinite(ll):
        return -np.inf
    return float(lp + ll)


# =============================================================================
# Numerical utilities
# =============================================================================


def central_hessian(fun: Callable[[np.ndarray], float], x: np.ndarray, step_rel: float = 1e-4) -> np.ndarray:
    """
    Central-difference Hessian for scalar objective.
    Uses parameter-scaled steps and returns a symmetrized matrix.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    f0 = float(fun(x))
    if not np.isfinite(f0):
        return np.full((n, n), np.nan)

    h = step_rel * np.maximum(1.0, np.abs(x))
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        fp = float(fun(xp))
        fm = float(fun(xm))
        if not (np.isfinite(fp) and np.isfinite(fm)):
            return np.full((n, n), np.nan)
        H[i, i] = (fp - 2.0 * f0 + fm) / (h[i] ** 2)

        for j in range(i + 1, n):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm[i] -= h[i]
            xmm[j] -= h[j]
            fpp = float(fun(xpp))
            fpm = float(fun(xpm))
            fmp = float(fun(xmp))
            fmm = float(fun(xmm))
            if not all(np.isfinite(v) for v in [fpp, fpm, fmp, fmm]):
                return np.full((n, n), np.nan)
            H_ij = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    H = 0.5 * (H + H.T)
    return H


def stabilized_logdet(H: np.ndarray, ridge_init: float = 1e-9, ridge_max: float = 1e6) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute log|H| for PD H, adding ridge if needed.
    Returns (logdet, ridge_used). If fails, returns (None, None).
    """
    if H is None or np.any(~np.isfinite(H)):
        return None, None
    Hs = 0.5 * (H + H.T)

    ridges = [0.0]
    r = ridge_init
    while r <= ridge_max:
        ridges.append(r)
        r *= 10.0

    I = np.eye(Hs.shape[0])
    for ridge in ridges:
        H_try = Hs + ridge * I
        try:
            eig = np.linalg.eigvalsh(H_try)
        except np.linalg.LinAlgError:
            continue
        if np.all(eig > 0.0):
            logdet = float(np.sum(np.log(eig)))
            return logdet, ridge
    return None, None


def stable_inverse(H: np.ndarray, ridge_init: float = 1e-9, ridge_max: float = 1e6) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Invert Hessian after optional ridge stabilization.
    Returns (cov, ridge_used), or (None, None) if unsuccessful.
    """
    if H is None or np.any(~np.isfinite(H)):
        return None, None
    Hs = 0.5 * (H + H.T)

    ridges = [0.0]
    r = ridge_init
    while r <= ridge_max:
        ridges.append(r)
        r *= 10.0

    I = np.eye(Hs.shape[0])
    for ridge in ridges:
        H_try = Hs + ridge * I
        try:
            eig = np.linalg.eigvalsh(H_try)
            if np.min(eig) <= 0.0:
                continue
            cov = np.linalg.inv(H_try)
            return cov, ridge
        except np.linalg.LinAlgError:
            continue
    return None, None


def calc_aic_bic(loglik: float, n_params: int, n_obs: int) -> Tuple[float, float]:
    """Return (AIC, BIC)."""
    aic = 2.0 * n_params - 2.0 * loglik
    bic = np.log(n_obs) * n_params - 2.0 * loglik
    return float(aic), float(bic)


# =============================================================================
# Optimization helpers
# =============================================================================


def random_start_in_bounds(bounds: Sequence[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """Uniform random start within box bounds."""
    vals = [rng.uniform(low, high) for (low, high) in bounds]
    return np.array(vals, dtype=float)


def multistart_lbfgsb(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    n_starts: int,
    seed: int,
    x0_candidates: Optional[List[np.ndarray]] = None,
    maxiter: int = 1000,
) -> optimize.OptimizeResult:
    """
    Multi-start L-BFGS-B minimization.
    Returns best finite result among all starts.
    """
    rng = np.random.default_rng(seed)
    starts: List[np.ndarray] = []
    if x0_candidates is not None:
        starts.extend([np.array(s, dtype=float) for s in x0_candidates])
    for _ in range(n_starts):
        starts.append(random_start_in_bounds(bounds, rng))

    best_res = None
    for x0 in starts:
        try:
            res = optimize.minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter},
            )
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    if best_res is None:
        raise RuntimeError("Optimization failed: no finite objective value found across starts.")
    return best_res


def heuristic_start_picm_l(data: ReliabilityData) -> np.ndarray:
    """Simple deterministic start for PICM-L."""
    med = []
    xs = []
    for s in data.stress_levels:
        d = data.df[data.df["delta_T"] == s]
        med.append(float(np.median(d["cycles"])))
        xs.append(float(np.log(s)))
    med = np.array(med, dtype=float)
    xs = np.array(xs, dtype=float)
    b, _a = np.polyfit(xs, np.log(med), deg=1)  # log(med) ~ a + b*x
    beta1_init = max(0.8, -b)
    beta0_init = float(np.mean(np.log(med) + beta1_init * xs))
    k_init = 3.0
    return np.array([k_init, beta1_init, beta0_init], dtype=float)


def heuristic_start_lognormal(data: ReliabilityData) -> np.ndarray:
    """Simple deterministic start for lognormal AFT."""
    med = []
    xs = []
    for s in data.stress_levels:
        d = data.df[data.df["delta_T"] == s]
        med.append(float(np.median(np.log(d["cycles"]))))
        xs.append(float(np.log(s)))
    med = np.array(med, dtype=float)
    xs = np.array(xs, dtype=float)
    b, _a = np.polyfit(xs, med, deg=1)
    beta1 = max(0.8, -b)
    beta0 = float(np.mean(med + beta1 * xs))
    sigma = 0.5
    return np.array([sigma, beta1, beta0], dtype=float)


# =============================================================================
# Frequentist fits (MLE, Firth, sensitivity models)
# =============================================================================


def fit_picm_l_mle(data: ReliabilityData, n_starts: int, seed: int) -> Dict[str, object]:
    """Fit PICM-L by standard MLE."""
    bounds = [(1.001, 25.0), (1e-4, 8.0), (-5.0, 30.0)]  # k, beta1, beta0
    start0 = heuristic_start_picm_l(data)
    start0 = np.clip(start0, [b[0] for b in bounds], [b[1] for b in bounds])
    penalty = 1e12

    def obj(theta: np.ndarray) -> float:
        v = nll_weibull_picm_l(theta, data)
        return float(v) if np.isfinite(v) else penalty

    res = multistart_lbfgsb(
        objective=obj,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        x0_candidates=[start0],
    )
    theta_hat = np.array(res.x, dtype=float)
    ll_hat = loglik_weibull_picm_l(theta_hat, data)

    H = central_hessian(lambda t: nll_weibull_picm_l(t, data), theta_hat)
    cov, ridge = stable_inverse(H)
    se = np.full(theta_hat.shape[0], np.nan)
    if cov is not None:
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    else:
        warnings.warn("MLE Hessian inversion failed; standard errors unavailable.")

    return {
        "theta": theta_hat,
        "loglik": ll_hat,
        "optim_result": res,
        "hessian": H,
        "cov": cov,
        "se": se,
        "ridge_for_cov": ridge,
    }


class FirthObjective:
    """Firth-penalized objective tracker for PICM-L."""

    def __init__(self, data: ReliabilityData):
        self.data = data
        self.ridge_events: List[float] = []
        self.non_pd_count = 0

    def __call__(self, theta: np.ndarray) -> float:
        nll = nll_weibull_picm_l(theta, self.data)
        if not np.isfinite(nll):
            return 1e12

        H = central_hessian(lambda t: nll_weibull_picm_l(t, self.data), np.array(theta, dtype=float))
        logdet, ridge = stabilized_logdet(H)
        if (logdet is None) or (ridge is None):
            self.non_pd_count += 1
            return 1e12 + nll
        if ridge > 0.0:
            self.ridge_events.append(ridge)
        # Minimize: nll - 0.5*log|I(theta)|
        return float(nll - 0.5 * logdet)


def fit_picm_l_firth(data: ReliabilityData, mle_theta: np.ndarray, n_starts: int, seed: int) -> Dict[str, object]:
    """Fit PICM-L by Firth-penalized MLE."""
    bounds = [(1.001, 25.0), (1e-4, 8.0), (-5.0, 30.0)]  # k, beta1, beta0
    firth_obj = FirthObjective(data)

    res = multistart_lbfgsb(
        objective=firth_obj,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        x0_candidates=[np.array(mle_theta, dtype=float), heuristic_start_picm_l(data)],
    )
    theta_hat = np.array(res.x, dtype=float)
    ll_at_hat = loglik_weibull_picm_l(theta_hat, data)

    # Hessian-based SE from penalized objective where stable
    H_pen = central_hessian(firth_obj, theta_hat)
    cov_pen, ridge = stable_inverse(H_pen)
    se_pen = np.full(theta_hat.shape[0], np.nan)
    if cov_pen is not None:
        se_pen = np.sqrt(np.clip(np.diag(cov_pen), 0.0, np.inf))
    else:
        warnings.warn("Firth Hessian inversion failed; standard errors unavailable.")

    return {
        "theta": theta_hat,
        "loglik": ll_at_hat,
        "optim_result": res,
        "hessian_pen": H_pen,
        "cov_pen": cov_pen,
        "se_pen": se_pen,
        "ridge_for_cov": ridge,
        "ridge_events": firth_obj.ridge_events,
        "non_pd_count": firth_obj.non_pd_count,
        "ridge_trigger_count": int(len(firth_obj.ridge_events)),
        "rejected_hessian_count": int(firth_obj.non_pd_count),
    }


def fit_picm_c_sensitivity(
    data: ReliabilityData,
    n_starts: int,
    seed: int,
    prior_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Fit PICM-C curvature sensitivity model with regularization on beta2 toward 0:
    penalty = 0.5*(beta2/beta2_sd)^2 (equivalent to N(0, beta2_sd^2) prior in MAP objective).
    """
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    beta2_sd = float(cfg["beta2_sd"])
    bounds = [(1.001, 25.0), (1e-4, 8.0), (-5.0, 30.0), (-2.0, 2.0)]  # k, beta1, beta0, beta2
    start_l = heuristic_start_picm_l(data)
    start0 = np.array([start_l[0], start_l[1], start_l[2], 0.0], dtype=float)

    def obj(theta: np.ndarray) -> float:
        nll = nll_weibull_picm_c(theta, data)
        if not np.isfinite(nll):
            return 1e12
        beta2 = float(theta[3])
        reg = 0.5 * (beta2 / beta2_sd) ** 2
        return nll + reg

    res = multistart_lbfgsb(
        objective=obj,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        x0_candidates=[start0],
    )
    theta_hat = np.array(res.x, dtype=float)
    ll_hat = loglik_weibull_picm_c(theta_hat, data)
    aic, bic = calc_aic_bic(ll_hat, n_params=4, n_obs=data.n)
    H = central_hessian(obj, theta_hat)
    cov, ridge = stable_inverse(H)
    if cov is None:
        warnings.warn("PICM-C Hessian inversion failed; Laplace covariance unavailable.")
    return {
        "theta": theta_hat,
        "loglik": ll_hat,
        "aic": aic,
        "bic": bic,
        "optim_result": res,
        "hessian_pen": H,
        "cov_pen": cov,
        "ridge_for_cov": ridge,
        "beta2_penalty_sd": beta2_sd,
    }


def fit_lognormal_aft(data: ReliabilityData, n_starts: int, seed: int) -> Dict[str, object]:
    """Fit censored lognormal AFT sensitivity model by MLE."""
    bounds = [(1e-3, 5.0), (1e-4, 8.0), (-5.0, 30.0)]  # sigma, beta1, beta0
    start0 = heuristic_start_lognormal(data)
    start0 = np.clip(start0, [b[0] for b in bounds], [b[1] for b in bounds])
    penalty = 1e12

    def obj(theta: np.ndarray) -> float:
        v = nll_lognormal_aft(theta, data)
        return float(v) if np.isfinite(v) else penalty

    res = multistart_lbfgsb(
        objective=obj,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        x0_candidates=[start0],
    )
    theta_hat = np.array(res.x, dtype=float)
    ll_hat = loglik_lognormal_aft(theta_hat, data)
    aic, bic = calc_aic_bic(ll_hat, n_params=3, n_obs=data.n)
    H = central_hessian(lambda t: nll_lognormal_aft(t, data), theta_hat)
    cov, ridge = stable_inverse(H)
    if cov is None:
        warnings.warn("Lognormal AFT Hessian inversion failed; Laplace covariance unavailable.")
    return {
        "theta": theta_hat,
        "loglik": ll_hat,
        "aic": aic,
        "bic": bic,
        "optim_result": res,
        "hessian": H,
        "cov": cov,
        "ridge_for_cov": ridge,
    }


# =============================================================================
# Adaptive Metropolis Bayesian inference
# =============================================================================


def make_picm_l_mcmc_starts(mle_theta: np.ndarray, prior_cfg: Optional[Dict[str, float]] = None) -> List[np.ndarray]:
    """Build three dispersed initial states for PICM-L MCMC respecting prior truncation."""
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    k_lower = float(cfg["k_trunc_lower"])
    b1_lower = float(cfg["beta1_trunc_lower"])
    k0, b10, b00 = [float(v) for v in mle_theta]
    starts = [
        np.array([max(k0, k_lower + 0.01), max(b10, b1_lower + 0.01), b00], dtype=float),
        np.array([max(k_lower + 0.05, k0 * 0.8), max(b1_lower + 0.05, b10 * 1.2), b00 - 1.2], dtype=float),
        np.array([max(k_lower + 0.05, k0 * 1.2), max(b1_lower + 0.05, b10 * 0.8), b00 + 1.2], dtype=float),
    ]
    return starts


def run_adaptive_mh_picm_l(
    data: ReliabilityData,
    starts: List[np.ndarray],
    seeds: List[int],
    burnin: int,
    keep: int,
    adapt_block: int = 100,
    prior_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Adaptive random-walk Metropolis for PICM-L:
    - adapt covariance during burn-in only
    - freeze kernel after burn-in
    - retain full post-burn chain
    """
    d = 3
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    k_lower = float(cfg["k_trunc_lower"])
    b1_lower = float(cfg["beta1_trunc_lower"])
    n_chains = len(starts)
    if len(seeds) != n_chains:
        raise ValueError("Number of seeds must match number of chains.")
    total = burnin + keep

    init_cov = np.diag([0.12**2, 0.08**2, 0.35**2])
    chains = np.zeros((n_chains, keep, d), dtype=float)
    accept_rates = np.zeros(n_chains, dtype=float)
    final_covs = []
    final_log_scales = []

    for c in range(n_chains):
        rng = np.random.default_rng(seeds[c])
        theta = np.array(starts[c], dtype=float)
        lp = logposterior_picm_l(theta, data, prior_cfg=cfg)
        if not np.isfinite(lp):
            raise RuntimeError(f"Invalid initial state for chain {c + 1}: {theta}")

        base_cov = init_cov.copy()
        log_scale = 0.0
        prop_cov = math.exp(2.0 * log_scale) * base_cov
        sample_accepted = 0
        accepted_states: List[np.ndarray] = []
        chain_all = np.zeros((total, d), dtype=float)

        for t in range(total):
            accepted_this = 0
            prop = rng.multivariate_normal(mean=theta, cov=prop_cov)
            # Reject invalid proposals immediately before costly likelihood eval.
            if (prop[0] > k_lower) and (prop[1] > b1_lower) and np.isfinite(prop[2]):
                lp_prop = logposterior_picm_l(prop, data, prior_cfg=cfg)
                if np.isfinite(lp_prop):
                    log_alpha = lp_prop - lp
                    if np.log(rng.uniform()) < log_alpha:
                        theta = prop
                        lp = lp_prop
                        accepted_this = 1

            chain_all[t] = theta

            if t < burnin:
                if accepted_this:
                    accepted_states.append(theta.copy())
                gamma_t = float((t + 1) ** -0.6)
                log_scale = float(np.clip(log_scale + gamma_t * (accepted_this - MCMC_TARGET_ACCEPT), -3.0, 3.0))
                if ((t + 1) % adapt_block == 0) and len(accepted_states) >= (d + 2):
                    arr = np.array(accepted_states, dtype=float)
                    emp_cov = np.cov(arr.T, ddof=1)
                    emp_cov = 0.5 * (emp_cov + emp_cov.T)
                    eig = np.linalg.eigvalsh(emp_cov)
                    min_eig = float(np.min(eig))
                    if min_eig <= 0.0:
                        emp_cov = emp_cov + (abs(min_eig) + 1e-8) * np.eye(d)
                    base_cov = (2.38**2 / d) * (emp_cov + 1e-8 * np.eye(d))
                    base_cov = 0.5 * (base_cov + base_cov.T)
                prop_cov = math.exp(2.0 * log_scale) * base_cov
                prop_cov = 0.5 * (prop_cov + prop_cov.T)
            else:
                sample_accepted += accepted_this

        chains[c] = chain_all[burnin:, :]
        accept_rates[c] = sample_accepted / keep
        final_covs.append(prop_cov.copy())
        final_log_scales.append(float(log_scale))

    return {
        "chains": chains,
        "accept_rates": accept_rates,
        "final_covs": final_covs,
        "final_log_scales": np.asarray(final_log_scales, dtype=float),
        "accept_rate_basis": "post_adaptation_sampling_phase",
    }


# =============================================================================
# MCMC diagnostics and posterior summaries
# =============================================================================


def hpd_interval(samples: np.ndarray, prob: float = 0.95) -> Tuple[float, float]:
    """Compute shortest HPD interval from 1D samples."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n < 2:
        return float(x[0]), float(x[0])
    m = int(np.floor(prob * n))
    if m < 1:
        return float(np.min(x)), float(np.max(x))
    widths = x[m:] - x[: n - m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + m])


def compute_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin R-hat.
    chains shape = (m, n, p)
    """
    m, n, p = chains.shape
    rhat = np.full(p, np.nan)
    if (m < 2) or (n < 2):
        return rhat

    for j in range(p):
        x = chains[:, :, j]
        chain_means = np.mean(x, axis=1)
        chain_vars = np.var(x, axis=1, ddof=1)
        W = float(np.mean(chain_vars))
        B = float(n * np.var(chain_means, ddof=1))
        if W <= 0.0:
            rhat[j] = np.nan
            continue
        var_hat = ((n - 1.0) / n) * W + (B / n)
        rhat[j] = math.sqrt(max(var_hat / W, 0.0))
    return rhat


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Fast autocorrelation using FFT."""
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return np.ones(n)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    fx = np.fft.rfft(x, n=n_fft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=n_fft)[:n]
    acov = acov / np.arange(n, 0, -1)
    return acov / acov[0]


def effective_sample_size(chains_param: np.ndarray) -> float:
    """
    Effective sample size via initial positive sequence.
    chains_param shape = (m, n)
    """
    m, n = chains_param.shape
    if n < 4:
        return float(m * n)

    acorr_list = []
    for c in range(m):
        ac = _autocorr_fft(chains_param[c])
        acorr_list.append(ac)
    acorr = np.mean(np.vstack(acorr_list), axis=0)

    s = 0.0
    t = 1
    while t + 1 < n:
        pair_sum = acorr[t] + acorr[t + 1]
        if pair_sum < 0.0:
            break
        s += pair_sum
        t += 2
    tau = 1.0 + 2.0 * s
    if tau <= 0.0:
        return float(m * n)
    ess = (m * n) / tau
    ess = min(ess, float(m * n))
    return float(max(1.0, ess))


def summarize_posterior(samples: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Posterior summary table with ETI and HPD intervals."""
    rows = []
    for j, name in enumerate(names):
        x = samples[:, j]
        q025, q50, q975 = np.quantile(x, [0.025, 0.5, 0.975])
        hpd_l, hpd_u = hpd_interval(x, prob=0.95)
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(x)),
                "median": float(q50),
                "sd": float(np.std(x, ddof=1)),
                "eti_2.5%": float(q025),
                "eti_97.5%": float(q975),
                "hpd_2.5%": float(hpd_l),
                "hpd_97.5%": float(hpd_u),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# WAIC and DIC
# =============================================================================


def compute_loglik_matrix(samples: np.ndarray, data: ReliabilityData, batch: int = 4000) -> np.ndarray:
    """Compute pointwise log-likelihood samples matrix SxN for PICM-L."""
    S = samples.shape[0]
    N = data.n
    out = np.zeros((S, N), dtype=float)
    for i0 in range(0, S, batch):
        i1 = min(S, i0 + batch)
        for i in range(i0, i1):
            out[i, :] = loglik_weibull_picm_l(samples[i, :], data, pointwise=True)
    return out


def compute_waic_dic(samples: np.ndarray, data: ReliabilityData) -> Dict[str, float]:
    """
    Compute WAIC (primary Bayesian metric) and DIC (supplementary).
    """
    ll_mat = compute_loglik_matrix(samples, data)
    S = ll_mat.shape[0]

    lppd_i = special.logsumexp(ll_mat, axis=0) - np.log(S)
    lppd = float(np.sum(lppd_i))
    p_waic = float(np.sum(np.var(ll_mat, axis=0, ddof=1)))
    waic = float(-2.0 * (lppd - p_waic))

    dev = -2.0 * np.sum(ll_mat, axis=1)
    d_bar = float(np.mean(dev))
    theta_bar = np.mean(samples, axis=0)
    d_hat = float(-2.0 * loglik_weibull_picm_l(theta_bar, data))
    p_dic = float(d_bar - d_hat)
    dic = float(d_hat + 2.0 * p_dic)
    return {"waic": waic, "p_waic": p_waic, "dic": dic, "p_dic": p_dic}


# =============================================================================
# Use-condition B10 and profile likelihood + Bartlett correction
# =============================================================================


def weibull_b_quantile(eta: np.ndarray, k: np.ndarray, p_fail: float) -> np.ndarray:
    """Weibull B-quantile (cycles) at failure probability p_fail."""
    p = float(p_fail)
    if not (0.0 < p < 1.0):
        raise ValueError("p_fail must be strictly between 0 and 1.")
    c = -np.log1p(-p)
    return np.asarray(eta, dtype=float) * (c ** (1.0 / np.asarray(k, dtype=float)))


def weibull_median(eta: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Weibull median life (cycles)."""
    return weibull_b_quantile(eta, k, p_fail=0.5)


def lognormal_b_quantile(mu: np.ndarray, sigma: np.ndarray, p_fail: float) -> np.ndarray:
    """Lognormal B-quantile (cycles) at failure probability p_fail."""
    p = float(p_fail)
    if not (0.0 < p < 1.0):
        raise ValueError("p_fail must be strictly between 0 and 1.")
    z = stats.norm.ppf(p)
    return np.exp(np.asarray(mu, dtype=float) + np.asarray(sigma, dtype=float) * z)


def lognormal_median(mu: np.ndarray) -> np.ndarray:
    """Lognormal median life (cycles)."""
    return np.exp(np.asarray(mu, dtype=float))


def weibull_quantile(eta: np.ndarray, k: np.ndarray, p: float) -> np.ndarray:
    """Compute the Weibull quantile at probability p."""
    return weibull_b_quantile(eta, k, p_fail=p)


def b10_use_from_theta(theta: np.ndarray, delta_t_use: float) -> float:
    """Use-condition B10 for PICM-L at delta_t_use."""
    k, beta1, beta0 = [float(v) for v in theta]
    x_use = np.log(delta_t_use)
    eta_use = np.exp(beta0 - beta1 * x_use)
    c = -np.log(0.9)
    return float(eta_use * (c ** (1.0 / k)))


def logpsi_from_theta(theta: np.ndarray, delta_t_use: float) -> float:
    """log(B10_use)."""
    psi = b10_use_from_theta(theta, delta_t_use)
    return float(np.log(psi))


def theta_from_nuisance_and_logpsi(k: float, beta1: float, logpsi: float, delta_t_use: float) -> np.ndarray:
    """
    Reconstruct full theta from nuisance (k, beta1) and fixed logpsi:
    logpsi = beta0 - beta1*x_use + log(c)/k
    => beta0 = logpsi + beta1*x_use - log(c)/k
    """
    c = -np.log(0.9)
    x_use = np.log(delta_t_use)
    beta0 = logpsi + beta1 * x_use - np.log(c) / k
    return np.array([k, beta1, beta0], dtype=float)


def fit_constrained_logpsi(
    data: ReliabilityData,
    logpsi: float,
    delta_t_use: float,
    seed: int,
    starts: Optional[List[np.ndarray]] = None,
) -> Dict[str, object]:
    """
    Constrained optimization under fixed log(B10_use), profiling over nuisance (k, beta1).
    """
    bounds_nuis = [(1.001, 25.0), (1e-4, 8.0)]  # k, beta1

    def obj_nuis(nuis: np.ndarray) -> float:
        k, beta1 = nuis
        if (k <= 1.0) or (beta1 <= 0.0):
            return 1e12
        theta = theta_from_nuisance_and_logpsi(k, beta1, logpsi, delta_t_use)
        v = nll_weibull_picm_l(theta, data)
        return float(v) if np.isfinite(v) else 1e12

    x0s = [] if starts is None else [np.array(s, dtype=float) for s in starts]
    if not x0s:
        x0s = [np.array([3.0, 2.0], dtype=float)]

    res = multistart_lbfgsb(
        objective=obj_nuis,
        bounds=bounds_nuis,
        n_starts=6,
        seed=seed,
        x0_candidates=x0s,
        maxiter=600,
    )
    k_hat, b1_hat = res.x
    theta_hat = theta_from_nuisance_and_logpsi(k_hat, b1_hat, logpsi, delta_t_use)
    ll_hat = loglik_weibull_picm_l(theta_hat, data)
    return {"theta": theta_hat, "loglik": ll_hat, "optim_result": res}


def find_profile_ci(logpsi_grid: np.ndarray, lr_grid: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Interpolate profile-likelihood CI endpoints on logpsi scale."""
    idx_min = int(np.argmin(lr_grid))

    def interp_x(x1, y1, x2, y2, y):
        if y2 == y1:
            return float(0.5 * (x1 + x2))
        return float(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    # Left crossing
    left = logpsi_grid[0]
    for i in range(idx_min - 1, -1, -1):
        if lr_grid[i] > threshold and lr_grid[i + 1] <= threshold:
            left = interp_x(logpsi_grid[i], lr_grid[i], logpsi_grid[i + 1], lr_grid[i + 1], threshold)
            break

    # Right crossing
    right = logpsi_grid[-1]
    for i in range(idx_min, len(logpsi_grid) - 1):
        if lr_grid[i] <= threshold and lr_grid[i + 1] > threshold:
            right = interp_x(logpsi_grid[i], lr_grid[i], logpsi_grid[i + 1], lr_grid[i + 1], threshold)
            break

    return left, right


def profile_likelihood_b10(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    mle_loglik: float,
    delta_t_use: float,
    n_grid: int,
    seed: int,
) -> Dict[str, object]:
    """Compute uncorrected profile LR curve and CI for log(B10_use)."""
    rng = np.random.default_rng(seed)
    logpsi_hat = logpsi_from_theta(mle_theta, delta_t_use)

    cov = stable_inverse(central_hessian(lambda t: nll_weibull_picm_l(t, data), mle_theta))[0]
    if cov is not None:
        k, b1, _b0 = mle_theta
        c = -np.log(0.9)
        x_use = np.log(delta_t_use)
        grad_logpsi = np.array([-np.log(c) / (k**2), -x_use, 1.0], dtype=float)
        var_logpsi = float(grad_logpsi @ cov @ grad_logpsi)
        se_logpsi = float(np.sqrt(max(var_logpsi, 1e-12)))
    else:
        se_logpsi = 0.35

    width = max(4.0 * se_logpsi, 0.75)
    q95 = stats.chi2.ppf(0.95, df=1)

    best = None
    for expand in range(5):
        w = width * (2**expand)
        grid = np.linspace(logpsi_hat - w, logpsi_hat + w, n_grid)
        ll_prof = np.full(n_grid, -np.inf)
        starts = [mle_theta[:2].copy()]
        for i, lg in enumerate(grid):
            seed_i = int(rng.integers(1, 2**31 - 1))
            cres = fit_constrained_logpsi(
                data=data,
                logpsi=float(lg),
                delta_t_use=delta_t_use,
                seed=seed_i,
                starts=starts,
            )
            ll_prof[i] = cres["loglik"]
            starts = [cres["theta"][:2].copy()]  # warm start continuation

        lr = 2.0 * (mle_loglik - ll_prof)
        idx_min = int(np.argmin(lr))
        left_ok = np.any(lr[:idx_min] > q95)
        right_ok = np.any(lr[idx_min + 1 :] > q95)
        best = {"grid": grid, "ll_prof": ll_prof, "lr": lr}
        if left_ok and right_ok:
            break

    left_logpsi, right_logpsi = find_profile_ci(best["grid"], best["lr"], q95)
    return {
        "logpsi_grid": best["grid"],
        "lr_grid": best["lr"],
        "ci_unc_logpsi": (left_logpsi, right_logpsi),
        "ci_unc_psi": (float(np.exp(left_logpsi)), float(np.exp(right_logpsi))),
        "q95": q95,
    }


def make_censor_limits_for_bootstrap(data: ReliabilityData) -> np.ndarray:
    """
    Build bootstrap censoring limits preserving observed censoring structure:
    - originally censored unit keeps its observed censor time
    - originally failed unit is treated as effectively uncensored (inf limit)
    """
    c = np.full(data.n, np.inf, dtype=float)
    cens_idx = np.where(data.event == 0)[0]
    c[cens_idx] = data.y[cens_idx]
    return c


def simulate_parametric_bootstrap(
    data: ReliabilityData,
    theta: np.ndarray,
    censor_limits: np.ndarray,
    rng: np.random.Generator,
) -> ReliabilityData:
    """Simulate bootstrap dataset under PICM-L with preserved censoring limits."""
    k, beta1, beta0 = [float(v) for v in theta]
    eta = np.exp(beta0 - beta1 * data.x_log)
    t = eta * rng.weibull(k, size=data.n)
    y_obs = np.minimum(t, censor_limits)
    ev = (t <= censor_limits).astype(int)

    df_b = data.df.copy()
    df_b["cycles"] = y_obs
    df_b["ttf_hours"] = y_obs * CYCLE_TIME_HOURS
    df_b["event"] = ev
    return prepare_data(df_b)


def bartlett_factor_bootstrap(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    logpsi_hat: float,
    delta_t_use: float,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    """
    Estimate Bartlett correction factor c_B from parametric bootstrap:
    c_B = E[LR] under approximate null / 1.
    """
    rng = np.random.default_rng(seed)
    censor_limits = make_censor_limits_for_bootstrap(data)

    lr_vals = []
    skipped = 0
    for _ in range(n_boot):
        db = simulate_parametric_bootstrap(data, mle_theta, censor_limits, rng)
        try:
            fit_u = fit_picm_l_mle(db, n_starts=N_OPT_STARTS_BOOT, seed=int(rng.integers(1, 2**31 - 1)))
            ll_u = fit_u["loglik"]

            starts_c = [fit_u["theta"][:2], mle_theta[:2]]
            fit_c = fit_constrained_logpsi(
                db,
                logpsi=float(logpsi_hat),
                delta_t_use=delta_t_use,
                seed=int(rng.integers(1, 2**31 - 1)),
                starts=starts_c,
            )
            ll_c = fit_c["loglik"]
            lr = 2.0 * (ll_u - ll_c)
            if np.isfinite(lr) and (lr >= 0):
                lr_vals.append(float(lr))
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if len(lr_vals) < max(20, int(0.6 * n_boot)):
        warnings.warn(
            f"Bartlett bootstrap had limited valid replicates: {len(lr_vals)}/{n_boot}. "
            "Correction may be noisy."
        )
    c_b = float(np.mean(lr_vals)) if lr_vals else 1.0
    c_b = max(c_b, 1e-8)
    return {"bartlett_factor": c_b, "lr_boot": np.array(lr_vals), "skipped": skipped}


# =============================================================================
# Posterior predictive quantities
# =============================================================================


def posterior_predictive_quantities(samples: np.ndarray, stress_levels: np.ndarray, delta_t_use: float) -> Dict[str, pd.DataFrame]:
    """Compute posterior summaries for median and B10 lives by stress + use condition."""
    k = samples[:, 0]
    b1 = samples[:, 1]
    b0 = samples[:, 2]

    def summarize(x: np.ndarray) -> Dict[str, float]:
        q025, q50, q975 = np.quantile(x, [0.025, 0.5, 0.975])
        return {
            "mean": float(np.mean(x)),
            "median": float(q50),
            "sd": float(np.std(x, ddof=1)),
            "eti_2.5%": float(q025),
            "eti_97.5%": float(q975),
        }

    rows_median = []
    rows_b10 = []
    for s in stress_levels:
        x = np.log(s)
        eta = np.exp(b0 - b1 * x)
        med = weibull_quantile(eta, k, 0.5)
        b10 = weibull_quantile(eta, k, 0.1)

        r_med = summarize(med)
        r_b10 = summarize(b10)
        r_med["delta_T"] = float(s)
        r_b10["delta_T"] = float(s)
        rows_median.append(r_med)
        rows_b10.append(r_b10)

    x_use = np.log(delta_t_use)
    eta_use = np.exp(b0 - b1 * x_use)
    use_med = weibull_quantile(eta_use, k, 0.5)
    use_b10 = weibull_quantile(eta_use, k, 0.1)
    use_df = pd.DataFrame(
        [
            {"quantity": "use_median_cycles", **summarize(use_med)},
            {"quantity": "use_B10_cycles", **summarize(use_b10)},
        ]
    )

    return {
        "stress_median": pd.DataFrame(rows_median).sort_values("delta_T").reset_index(drop=True),
        "stress_b10": pd.DataFrame(rows_b10).sort_values("delta_T").reset_index(drop=True),
        "use_summary": use_df,
        "use_b10_samples": use_b10,
        "use_median_samples": use_med,
    }


def _as_2d_draws(theta_or_samples: np.ndarray, n_params: int, name: str) -> np.ndarray:
    """Coerce a parameter vector or matrix into an SxP draw matrix."""
    arr = np.asarray(theta_or_samples, dtype=float)
    if arr.ndim == 1:
        if arr.size != n_params:
            raise ValueError(f"{name} expected length {n_params}, got {arr.size}.")
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        if arr.shape[1] != n_params:
            raise ValueError(f"{name} expected shape (S,{n_params}), got {arr.shape}.")
        return arr
    raise ValueError(f"{name} must be 1D or 2D array; got ndim={arr.ndim}.")


def _validate_delta_t_values(delta_t_values: np.ndarray) -> np.ndarray:
    """Validate and return 1D positive DeltaT array."""
    d = np.asarray(delta_t_values, dtype=float).reshape(-1)
    if d.size == 0:
        raise ValueError("delta_t_values must be non-empty.")
    if np.any(~np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("delta_t_values must contain finite positive values.")
    return d


def _summarize_draw_matrix(draws: np.ndarray, delta_t_values: np.ndarray, value_name: str) -> pd.DataFrame:
    """Summarize SxD draw matrix into quantiles by DeltaT."""
    q025, q50, q975 = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame(
        {
            "delta_T": delta_t_values.astype(float),
            f"{value_name}_mean": np.mean(draws, axis=0),
            f"{value_name}_q025": q025,
            f"{value_name}_q50": q50,
            f"{value_name}_q975": q975,
        }
    )


def predict_picm_l_quantities(theta_or_samples: np.ndarray, delta_t_values: np.ndarray) -> Dict[str, object]:
    """Predict PICM-L B10 and median life for arbitrary DeltaT values."""
    draws = _as_2d_draws(theta_or_samples, n_params=3, name="PICM-L parameters")
    delta = _validate_delta_t_values(delta_t_values)
    x = np.log(delta)[None, :]
    k = draws[:, 0][:, None]
    b1 = draws[:, 1][:, None]
    b0 = draws[:, 2][:, None]
    eta = np.exp(b0 - b1 * x)
    b10 = weibull_b_quantile(eta, k, p_fail=0.1)
    med = weibull_median(eta, k)
    return {
        "delta_t_values": delta,
        "b10_samples": b10,
        "median_samples": med,
        "b10_summary": _summarize_draw_matrix(b10, delta, "b10"),
        "median_summary": _summarize_draw_matrix(med, delta, "medianlife"),
    }


def predict_picm_c_quantities(theta_or_samples: np.ndarray, delta_t_values: np.ndarray, xbar: float) -> Dict[str, object]:
    """Predict PICM-C B10 and median life for arbitrary DeltaT values."""
    draws = _as_2d_draws(theta_or_samples, n_params=4, name="PICM-C parameters")
    delta = _validate_delta_t_values(delta_t_values)
    x = np.log(delta)[None, :]
    k = draws[:, 0][:, None]
    b1 = draws[:, 1][:, None]
    b0 = draws[:, 2][:, None]
    b2 = draws[:, 3][:, None]
    eta = np.exp(b0 - b1 * x + b2 * (x - float(xbar)) ** 2)
    b10 = weibull_b_quantile(eta, k, p_fail=0.1)
    med = weibull_median(eta, k)
    return {
        "delta_t_values": delta,
        "b10_samples": b10,
        "median_samples": med,
        "b10_summary": _summarize_draw_matrix(b10, delta, "b10"),
        "median_summary": _summarize_draw_matrix(med, delta, "medianlife"),
    }


def predict_lognormal_quantities(theta_or_samples: np.ndarray, delta_t_values: np.ndarray) -> Dict[str, object]:
    """Predict lognormal AFT B10 and median life for arbitrary DeltaT values."""
    draws = _as_2d_draws(theta_or_samples, n_params=3, name="Lognormal AFT parameters")
    delta = _validate_delta_t_values(delta_t_values)
    x = np.log(delta)[None, :]
    sigma = draws[:, 0][:, None]
    b1 = draws[:, 1][:, None]
    b0 = draws[:, 2][:, None]
    mu = b0 - b1 * x
    b10 = lognormal_b_quantile(mu, sigma, p_fail=0.1)
    med = lognormal_median(mu)
    return {
        "delta_t_values": delta,
        "b10_samples": b10,
        "median_samples": med,
        "b10_summary": _summarize_draw_matrix(b10, delta, "b10"),
        "median_summary": _summarize_draw_matrix(med, delta, "medianlife"),
    }


def sample_laplace_draws(
    mean: np.ndarray,
    cov: np.ndarray,
    n_draws: int,
    seed: int,
    valid_mask_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    """
    Draw from asymptotic normal approximation with rejection to enforce admissible domains.
    valid_mask_fn must return a boolean mask over rows of candidate draws.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    mu = np.asarray(mean, dtype=float)
    C = np.asarray(cov, dtype=float)
    if C.shape != (mu.size, mu.size):
        raise ValueError(f"cov shape {C.shape} incompatible with mean length {mu.size}.")

    rng = np.random.default_rng(seed)
    accepted: List[np.ndarray] = []
    accepted_n = 0
    attempts = 0
    batch = min(max(1000, n_draws // 2), 20000)
    max_attempts = max(20000, 250 * n_draws)

    while (accepted_n < n_draws) and (attempts < max_attempts):
        cand = rng.multivariate_normal(mean=mu, cov=C, size=batch)
        attempts += batch
        mask = valid_mask_fn(cand)
        if np.any(mask):
            acc = cand[mask]
            accepted.append(acc)
            accepted_n += acc.shape[0]
        # Adapt batch upward when acceptance is low.
        if accepted_n < n_draws // 4:
            batch = min(int(batch * 1.2), 50000)

    if accepted_n < n_draws:
        raise RuntimeError(
            f"Insufficient valid Laplace draws: requested {n_draws}, got {accepted_n} after {attempts} proposals."
        )

    draws = np.vstack(accepted)[:n_draws, :]
    return {
        "draws": draws,
        "n_draws": int(draws.shape[0]),
        "attempted_proposals": int(attempts),
        "acceptance_ratio_proposals": float(draws.shape[0] / max(attempts, 1)),
    }


def draw_picm_c_laplace(fit_picm_c: Dict[str, object], n_draws: int, seed: int) -> Dict[str, object]:
    """Generate PICM-C uncertainty draws via Laplace approximation (penalized objective curvature)."""
    theta = np.asarray(fit_picm_c["theta"], dtype=float)
    cov = fit_picm_c.get("cov_pen")
    method_note = "laplace_asymptotic_penalized"
    if cov is None:
        warnings.warn("PICM-C covariance unavailable; using conservative diagonal fallback for Laplace draws.")
        scale = np.maximum(np.abs(theta), 1.0) * 0.08
        cov = np.diag(scale**2)
        method_note = "laplace_diagonal_fallback"
    cov = np.asarray(cov, dtype=float)

    def valid_mask(arr: np.ndarray) -> np.ndarray:
        return (
            np.all(np.isfinite(arr), axis=1)
            & (arr[:, 0] > 1.0)  # k
            & (arr[:, 1] > 0.0)  # beta1
        )

    out = sample_laplace_draws(theta, cov, n_draws=n_draws, seed=seed, valid_mask_fn=valid_mask)
    out["method"] = method_note
    out["theta_center"] = theta
    out["covariance"] = cov
    return out


def draw_lognormal_laplace(fit_lognorm: Dict[str, object], n_draws: int, seed: int) -> Dict[str, object]:
    """Generate lognormal-AFT uncertainty draws via Laplace approximation."""
    theta = np.asarray(fit_lognorm["theta"], dtype=float)
    cov = fit_lognorm.get("cov")
    method_note = "laplace_asymptotic_nll"
    if cov is None:
        warnings.warn("Lognormal AFT covariance unavailable; using conservative diagonal fallback for Laplace draws.")
        scale = np.maximum(np.abs(theta), 1.0) * 0.08
        cov = np.diag(scale**2)
        method_note = "laplace_diagonal_fallback"
    cov = np.asarray(cov, dtype=float)

    def valid_mask(arr: np.ndarray) -> np.ndarray:
        return (
            np.all(np.isfinite(arr), axis=1)
            & (arr[:, 0] > 0.0)  # sigma
            & (arr[:, 1] > 0.0)  # beta1
        )

    out = sample_laplace_draws(theta, cov, n_draws=n_draws, seed=seed, valid_mask_fn=valid_mask)
    out["method"] = method_note
    out["theta_center"] = theta
    out["covariance"] = cov
    return out


def prob_survive_mission(
    samples_or_draws: np.ndarray,
    delta_t_values: np.ndarray,
    mission_cycles: float,
    model_name: str,
    xbar: Optional[float] = None,
) -> Dict[str, object]:
    """
    Estimate P(TTF > mission_cycles | DeltaT, model) from uncertainty draws.
    Supported model_name: 'picm_l', 'picm_c', 'lognormal'.
    """
    if mission_cycles <= 0:
        raise ValueError("mission_cycles must be > 0.")
    delta = _validate_delta_t_values(delta_t_values)
    x = np.log(delta)[None, :]
    m = float(mission_cycles)
    name = model_name.strip().lower()

    if name == "picm_l":
        draws = _as_2d_draws(samples_or_draws, n_params=3, name="PICM-L draws")
        k = draws[:, 0][:, None]
        b1 = draws[:, 1][:, None]
        b0 = draws[:, 2][:, None]
        eta = np.exp(b0 - b1 * x)
        surv = np.exp(-((m / eta) ** k))
    elif name == "picm_c":
        if xbar is None:
            raise ValueError("xbar is required for PICM-C mission survival.")
        draws = _as_2d_draws(samples_or_draws, n_params=4, name="PICM-C draws")
        k = draws[:, 0][:, None]
        b1 = draws[:, 1][:, None]
        b0 = draws[:, 2][:, None]
        b2 = draws[:, 3][:, None]
        eta = np.exp(b0 - b1 * x + b2 * (x - float(xbar)) ** 2)
        surv = np.exp(-((m / eta) ** k))
    elif name in {"lognormal", "lognormal_aft"}:
        draws = _as_2d_draws(samples_or_draws, n_params=3, name="Lognormal draws")
        sigma = draws[:, 0][:, None]
        b1 = draws[:, 1][:, None]
        b0 = draws[:, 2][:, None]
        mu = b0 - b1 * x
        surv = stats.norm.sf(np.log(m), loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'.")

    q025, q50, q975 = np.quantile(surv, [0.025, 0.5, 0.975], axis=0)
    summary = pd.DataFrame(
        {
            "delta_T": delta,
            "mission_cycles": m,
            "p_survive_mean": np.mean(surv, axis=0),
            "p_survive_q025": q025,
            "p_survive_q50": q50,
            "p_survive_q975": q975,
        }
    )
    return {"survival_samples": surv, "summary": summary}


def build_low_delta_modelwise_table(
    target_delta_ts: np.ndarray,
    pred_picm_l: Dict[str, object],
    pred_picm_c: Dict[str, object],
    pred_lognorm: Dict[str, object],
    mission_picm_l_1: pd.DataFrame,
    mission_picm_l_2: pd.DataFrame,
    mission_picm_c_1: pd.DataFrame,
    mission_picm_c_2: pd.DataFrame,
    mission_lognorm_1: pd.DataFrame,
    mission_lognorm_2: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-model low-DeltaT decision table."""
    delta = _validate_delta_t_values(target_delta_ts)

    specs = [
        ("PICM-L (Bayesian)", "MCMC posterior sampling", pred_picm_l, mission_picm_l_1, mission_picm_l_2),
        ("PICM-C (Laplace)", "Penalized fit + Laplace approximation", pred_picm_c, mission_picm_c_1, mission_picm_c_2),
        ("Lognormal AFT (Laplace)", "MLE fit + Laplace approximation", pred_lognorm, mission_lognorm_1, mission_lognorm_2),
    ]
    rows = []
    for model_name, uncertainty_method, pred, m1, m2 in specs:
        b10 = pred["b10_summary"].set_index("delta_T")
        med = pred["median_summary"].set_index("delta_T")
        ms1 = m1.set_index("delta_T")
        ms2 = m2.set_index("delta_T")
        for d in delta:
            rows.append(
                {
                    "model": model_name,
                    "uncertainty_method": uncertainty_method,
                    "delta_T": float(d),
                    "b10_q50": float(b10.loc[d, "b10_q50"]),
                    "b10_q025": float(b10.loc[d, "b10_q025"]),
                    "b10_q975": float(b10.loc[d, "b10_q975"]),
                    "medianlife_q50": float(med.loc[d, "medianlife_q50"]),
                    "medianlife_q025": float(med.loc[d, "medianlife_q025"]),
                    "medianlife_q975": float(med.loc[d, "medianlife_q975"]),
                    "p_survive_mission_1": float(ms1.loc[d, "p_survive_mean"]),
                    "p_survive_mission_2": float(ms2.loc[d, "p_survive_mean"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["delta_T", "model"]).reset_index(drop=True)


def build_low_delta_envelope_table(modelwise_table: pd.DataFrame) -> pd.DataFrame:
    """Build model-form envelope table across PICM-L, PICM-C and lognormal AFT."""
    rows = []
    for d, grp in modelwise_table.groupby("delta_T"):
        rows.append(
            {
                "delta_T": float(d),
                "b10_lower95_min": float(np.min(grp["b10_q025"])),
                "b10_upper95_max": float(np.max(grp["b10_q975"])),
                "b10_median_min": float(np.min(grp["b10_q50"])),
                "b10_median_max": float(np.max(grp["b10_q50"])),
                "medianlife_lower95_min": float(np.min(grp["medianlife_q025"])),
                "medianlife_upper95_max": float(np.max(grp["medianlife_q975"])),
                "medianlife_median_min": float(np.min(grp["medianlife_q50"])),
                "medianlife_median_max": float(np.max(grp["medianlife_q50"])),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_T").reset_index(drop=True)


def build_mission_survival_table(
    mission_tables: Dict[str, Dict[float, pd.DataFrame]],
) -> pd.DataFrame:
    """Flatten mission-survival summaries into one machine-readable table."""
    rows = []
    for model_name, by_mission in mission_tables.items():
        for mission_cycles, df in by_mission.items():
            for _, r in df.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "mission_cycles": float(mission_cycles),
                        "delta_T": float(r["delta_T"]),
                        "p_survive_mean": float(r["p_survive_mean"]),
                        "p_survive_q025": float(r["p_survive_q025"]),
                        "p_survive_q50": float(r["p_survive_q50"]),
                        "p_survive_q975": float(r["p_survive_q975"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["mission_cycles", "delta_T", "model"]).reset_index(drop=True)


# =============================================================================
# Additional diagnostics and submission-support outputs
# =============================================================================


PRIOR_SENSITIVITY_NOTE = (
    "Prior sensitivity is evaluated conditional on the PICM-L stress-life form; "
    "structural model-form uncertainty remains separate."
)

PRIOR_VS_DATA_NOTE = (
    "Posterior narrowing reflects information propagated through the assumed PICM-L "
    "stress-life relation; the prior-only and data-informed medians coincide closely. "
    "This is uncertainty reduction and prior-data corroboration, not direct low-stress validation."
)

GROUP_SHAPE_NOTE = (
    "The group-specific shapes are estimated from n=8 complete failures each; the "
    "maximum-likelihood shape is positively biased at this sample size. The intervals are "
    "wide and mutually overlapping, and a likelihood-ratio test does not provide significant "
    "evidence for stress-specific shapes. The common shape is retained for parsimony and "
    "identifiability, not as proof of physically common dispersion."
)

SCATTER_100C_NOTE = (
    "The observed 100 C range (720-2818 h) lies within the fitted single-population Weibull "
    "2.5-97.5% interval; the spread is compatible with ordinary Weibull dispersion and does "
    "not identify the internal physical mechanism."
)

FAILURE_ONLY_WARNING = (
    "Descriptive over observed failures only; NOT a primary metric for right-censored inference."
)

HIGH_DT_WARNING = (
    "Above tested range; may be invalid if new mechanisms or rating limits activate."
)

LOW_STRESS_DURATION_NOTE = (
    "Duration equivalence illustrates the feasibility burden of direct low-stress B10-scale "
    "validation; it is not a recommendation to omit validation."
)

MODEL_COMPARISON_NOTE = (
    "The lognormal AFT model is marginally better by AIC/BIC, with DeltaAIC < 2. "
    "The data do not decisively distinguish the Weibull and lognormal lifetime laws. "
    "PICM-L is retained as the primary reporting model for interpretability and "
    "regime-aware reporting, not because it decisively dominates by information criteria."
)

DECISION_BOUNDARY_NOTE = (
    "The crossings are posterior decision summaries conditional on PICM-L, not standalone "
    "model-free acceptance criteria. The spread of crossing temperatures shows uncertainty in the boundary."
)

FIRTH_NOTE = (
    "The Firth-type estimate is retained as a small-sample likelihood-route consistency check. "
    "Its agreement with the ordinary MLE indicates that the primary point estimates are not "
    "materially changed by this penalisation route. Bayesian posterior inference remains the "
    "main uncertainty-propagation mechanism."
)


def classify_evidential_regime(delta_t_c: float) -> str:
    """Return the evidential-regime label for a DeltaT value."""
    d = float(delta_t_c)
    tested = np.any(np.isclose(EXPECTED_STRESS_LEVELS, d))
    if d < OBSERVED_DOMAIN_MIN:
        return "extrapolative model-conditional"
    if tested and d <= LOW_INFO_TESTED_MAX:
        return "tested survival-constrained"
    if d < LOWEST_FAILURE_OBSERVED_STRESS:
        return "near-boundary model-conditional"
    if d <= OBSERVED_DOMAIN_MAX:
        return "failure-supported"
    return "extrapolative model-conditional"


def sample_picm_l_prior(n_draws: int, prior_key: str = "baseline", seed: int = 20260301) -> pd.DataFrame:
    """
    Sample k, beta1 and beta0 from the stated PICM-L priors only.

    No likelihood is used. The same support restrictions used by the posterior are applied:
    k > k_trunc_lower and beta1 > beta1_trunc_lower.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    cfg = get_prior_config(prior_key)
    rng = np.random.default_rng(seed)

    k_cdf_lower = stats.gamma.cdf(
        cfg["k_trunc_lower"],
        a=cfg["k_gamma_shape"],
        scale=1.0 / cfg["k_gamma_rate"],
    )
    u = rng.uniform(k_cdf_lower, 1.0, size=int(n_draws))
    k = stats.gamma.ppf(u, a=cfg["k_gamma_shape"], scale=1.0 / cfg["k_gamma_rate"])

    beta1_a = (cfg["beta1_trunc_lower"] - cfg["beta1_mean"]) / cfg["beta1_sd"]
    beta1 = stats.truncnorm.rvs(
        beta1_a,
        np.inf,
        loc=cfg["beta1_mean"],
        scale=cfg["beta1_sd"],
        size=int(n_draws),
        random_state=rng,
    )
    beta0 = rng.normal(cfg["beta0_mean"], cfg["beta0_sd"], size=int(n_draws))

    return pd.DataFrame(
        {
            "draw_id": np.arange(int(n_draws), dtype=int),
            "prior_key": prior_key,
            "k": k.astype(float),
            "beta1": beta1.astype(float),
            "beta0": beta0.astype(float),
        }
    )


def build_prior_sensitivity_b10_extended(
    draws_by_prior: Dict[str, np.ndarray],
    delta_targets: np.ndarray = PRIOR_SENS_DELTA_TARGETS,
) -> pd.DataFrame:
    """Build prior-sensitivity B10 table at 20, 25 and 35 C."""
    targets = _validate_delta_t_values(delta_targets)
    rows: List[Dict[str, object]] = []
    summaries: Dict[Tuple[str, float], Tuple[float, float, float, float]] = {}

    for key, draws in draws_by_prior.items():
        pred = predict_picm_l_quantities(np.asarray(draws, dtype=float), targets)
        b10 = np.asarray(pred["b10_samples"], dtype=float)
        for j, d in enumerate(targets):
            q50, q025, q975 = _summarize_interval(b10[:, j])
            width = float(q975 - q025)
            summaries[(key, float(d))] = (q025, q50, q975, width)

    for key in PRIOR_SENSITIVITY_KEYS:
        if key not in draws_by_prior:
            continue
        for d in targets:
            q025, q50, q975, width = summaries[(key, float(d))]
            base_q025, base_q50, base_q975, base_width = summaries[(BASELINE_PRIOR_KEY, float(d))]
            rows.append(
                {
                    "prior_setting": key,
                    "delta_T_C": float(d),
                    "B10_q2_5": q025,
                    "B10_q50": q50,
                    "B10_q97_5": q975,
                    "median_shift_percent_vs_baseline": 100.0 * (q50 - base_q50) / base_q50,
                    "width_ratio_vs_baseline": width / base_width if base_width > 0 else np.nan,
                    "evidential_regime": classify_evidential_regime(float(d)),
                    "interpretation_note": PRIOR_SENSITIVITY_NOTE,
                }
            )
    return pd.DataFrame(rows)


def build_prior_vs_data_b10(
    posterior_samples: np.ndarray,
    prior_draws: pd.DataFrame,
    delta_targets: np.ndarray = PRIOR_VS_DATA_TARGETS,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compare prior-only and data-informed PICM-L B10 projections."""
    targets = _validate_delta_t_values(delta_targets)
    prior_theta = prior_draws[["k", "beta1", "beta0"]].to_numpy(dtype=float)
    post_theta = np.asarray(posterior_samples, dtype=float)

    prior_b10 = np.asarray(predict_picm_l_quantities(prior_theta, targets)["b10_samples"], dtype=float)
    post_b10 = np.asarray(predict_picm_l_quantities(post_theta, targets)["b10_samples"], dtype=float)

    rows: List[Dict[str, object]] = []
    for j, d in enumerate(targets):
        prior_q50, prior_q025, prior_q975 = _summarize_interval(prior_b10[:, j])
        post_q50, post_q025, post_q975 = _summarize_interval(post_b10[:, j])
        prior_width = float(prior_q975 - prior_q025)
        post_width = float(post_q975 - post_q025)
        rows.append(
            {
                "delta_T_C": float(d),
                "prior_q2_5": prior_q025,
                "prior_q50": prior_q50,
                "prior_q97_5": prior_q975,
                "post_q2_5": post_q025,
                "post_q50": post_q50,
                "post_q97_5": post_q975,
                "median_ratio_post_over_prior": post_q50 / prior_q50 if prior_q50 > 0 else np.nan,
                "width_ratio_prior_over_post": prior_width / post_width if post_width > 0 else np.nan,
                "evidential_regime": classify_evidential_regime(float(d)),
                "interpretation_note": PRIOR_VS_DATA_NOTE,
            }
        )
    return pd.DataFrame(rows), prior_b10, post_b10


def save_simple_figure(fig: plt.Figure, out_dir: Path, basename: str) -> None:
    """Save a figure through the canonical publication figure writer."""
    group = _figure_group_from_dir(out_dir)
    save_pub_figure(fig, basename, group, columns=_infer_columns_from_figure(fig))


def plot_prior_vs_data_b10(
    table: pd.DataFrame,
    out_dir: Path,
    basename: str = "fig_prior_vs_data_b10",
) -> None:
    """Plot prior-only and data-informed B10 intervals."""
    tab = table.sort_values("delta_T_C").reset_index(drop=True)
    x = np.arange(len(tab), dtype=float)
    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.85))

    prior_q50 = tab["prior_q50"].to_numpy(dtype=float)
    prior_lo = tab["prior_q2_5"].to_numpy(dtype=float)
    prior_hi = tab["prior_q97_5"].to_numpy(dtype=float)
    post_q50 = tab["post_q50"].to_numpy(dtype=float)
    post_lo = tab["post_q2_5"].to_numpy(dtype=float)
    post_hi = tab["post_q97_5"].to_numpy(dtype=float)

    ax.errorbar(
        x - 0.08,
        prior_q50,
        yerr=[prior_q50 - prior_lo, prior_hi - prior_q50],
        fmt="o",
        color=PLOT_COLORS["km_empirical"],
        ecolor=PLOT_COLORS["km_empirical"],
        capsize=3,
        lw=1.3,
        label="Prior-only",
    )
    ax.errorbar(
        x + 0.08,
        post_q50,
        yerr=[post_q50 - post_lo, post_hi - post_q50],
        fmt="s",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        capsize=3,
        lw=1.3,
        label="Data-informed posterior",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}" for d in tab["delta_T_C"]])
    ax.set_yscale("log")
    ax.set_xlabel("DeltaT (C)")
    ax.set_ylabel("B10 life (cycles)")
    ax.set_title("Prior-only and data-informed B10 projections")
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=2,
        fontsize=8.0,
        rect_top=0.88,
        legend_y=0.985,
        columnspacing=1.2,
        handlelength=1.8,
    )
    save_simple_figure(fig, out_dir, basename)
    plt.close(fig)


def weibull_complete_loglik(y: np.ndarray, k: float, eta: float) -> float:
    """Complete-data two-parameter Weibull log-likelihood with loc fixed at zero."""
    yy = np.asarray(y, dtype=float)
    if yy.size == 0 or k <= 0 or eta <= 0 or np.any(yy <= 0):
        return float("-inf")
    return float(np.sum(np.log(k) + (k - 1.0) * np.log(yy) - k * np.log(eta) - (yy / eta) ** k))


def group_shape_common_k_lrt(
    data: ReliabilityData,
    stress_values: Sequence[float] = (100.0, 125.0, 150.0),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Likelihood-ratio test for common versus stress-specific Weibull shape."""
    groups: List[np.ndarray] = []
    for d in stress_values:
        grp = data.df[np.isclose(data.df["delta_T"].to_numpy(dtype=float), float(d))].copy()
        if not bool(np.all(grp["event"].to_numpy(dtype=int) == 1)):
            raise ValueError(f"Stress {d:g} C is not a complete-failure group.")
        y = grp["cycles"].to_numpy(dtype=float)
        if y.size < 2:
            raise ValueError(f"Stress {d:g} C has too few failures for group-shape LRT.")
        groups.append(y)

    loglik_full = 0.0
    for y in groups:
        k_g, _, eta_g = stats.weibull_min.fit(y, floc=0)
        loglik_full += weibull_complete_loglik(y, float(k_g), float(eta_g))

    def neg_common(log_k_arr: np.ndarray) -> float:
        k_common = float(np.exp(log_k_arr[0]))
        ll = 0.0
        for y in groups:
            eta_hat = float(np.mean(y ** k_common) ** (1.0 / k_common))
            ll += weibull_complete_loglik(y, k_common, eta_hat)
        return -ll

    opt = optimize.minimize(neg_common, x0=np.array([math.log(3.0)]), method="Nelder-Mead")
    if not opt.success:
        opt = optimize.minimize(neg_common, x0=np.array([math.log(3.0)]), method="BFGS")
    common_k_hat = float(np.exp(opt.x[0]))
    loglik_common = float(-opt.fun)
    lr_stat = float(2.0 * (loglik_full - loglik_common))
    df_lrt = 2
    p_value = float(stats.chi2.sf(max(lr_stat, 0.0), df_lrt))
    if p_value >= alpha:
        conclusion = (
            f"No significant evidence against a common Weibull shape among the fully failed groups "
            f"(LRT, df=2, p={p_value:.3f}); common k retained for parsimony and identifiability."
        )
    else:
        conclusion = (
            f"LRT indicates some evidence of shape heterogeneity (p={p_value:.3f}); common k retained "
            "as a deliberate identifiability assumption under sparse data, with this caveat stated."
        )

    return pd.DataFrame(
        [
            {
                "loglik_common_k": loglik_common,
                "loglik_pergroup_k": float(loglik_full),
                "lr_stat": lr_stat,
                "df": df_lrt,
                "p_value": p_value,
                "common_k_hat": common_k_hat,
                "alpha": float(alpha),
                "conclusion": conclusion,
            }
        ]
    )


def fit_group_specific_weibull_complete(
    data: ReliabilityData,
    pooled_picm_l_k: float,
    stress_values: Sequence[float] = (100.0, 125.0, 150.0),
    n_boot: int = N_GROUP_SHAPE_BOOT,
    seed: int = SEED_BOOT + 900,
) -> pd.DataFrame:
    """Fit two-parameter Weibull distributions to complete-failure groups."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for d in stress_values:
        grp = data.df[np.isclose(data.df["delta_T"].to_numpy(dtype=float), float(d))].copy()
        fail_cycles = grp.loc[grp["event"].astype(int) == 1, "cycles"].to_numpy(dtype=float)
        n_total = int(len(grp))
        n_failed = int(len(fail_cycles))
        boot_bias = k_hat_bc = ci_bc_lo = ci_bc_hi = np.nan
        n_boot_effective = 0
        if n_failed < 2:
            k_hat = eta_hat = ci_lo = ci_hi = np.nan
        else:
            k_hat, _, eta_hat = stats.weibull_min.fit(fail_cycles, floc=0)
            boot_k: List[float] = []
            for _ in range(int(n_boot)):
                sample = rng.choice(fail_cycles, size=n_failed, replace=True)
                try:
                    k_b, _, _ = stats.weibull_min.fit(sample, floc=0)
                    if np.isfinite(k_b):
                        boot_k.append(float(k_b))
                except Exception:
                    continue
            if boot_k:
                boot_arr = np.asarray(boot_k, dtype=float)
                n_boot_effective = int(boot_arr.size)
                boot_bias = float(np.mean(boot_arr) - float(k_hat))
                k_hat_bc = float(k_hat - boot_bias)
                boot_bc = boot_arr - boot_bias
                ci_lo, ci_hi = np.quantile(boot_arr, [0.025, 0.975])
                ci_bc_lo, ci_bc_hi = np.quantile(boot_bc, [0.025, 0.975])
            else:
                ci_lo = ci_hi = np.nan
        pooled_inside = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= pooled_picm_l_k <= ci_hi)
        pooled_inside_bc = bool(np.isfinite(ci_bc_lo) and np.isfinite(ci_bc_hi) and ci_bc_lo <= pooled_picm_l_k <= ci_bc_hi)
        rows.append(
            {
                "delta_T_C": float(d),
                "n_total": n_total,
                "n_failed": n_failed,
                "k_hat": float(k_hat),
                "eta_hat_cycles": float(eta_hat),
                "k_ci_lower": float(ci_lo),
                "k_ci_upper": float(ci_hi),
                "bootstrap_bias": float(boot_bias),
                "k_hat_bias_corrected": float(k_hat_bc),
                "k_ci_bias_corrected_lower": float(ci_bc_lo),
                "k_ci_bias_corrected_upper": float(ci_bc_hi),
                "pooled_picm_l_k": float(pooled_picm_l_k),
                "pooled_k_inside_ci": pooled_inside,
                "pooled_k_inside_ci_bias_corrected": pooled_inside_bc,
                "bootstrap_seed": int(seed),
                "n_boot_requested": int(n_boot),
                "n_boot_effective": n_boot_effective,
                "interpretation_note": GROUP_SHAPE_NOTE,
            }
        )
    return pd.DataFrame(rows)


def plot_group_shape_sensitivity(
    table: pd.DataFrame,
    out_dir: Path,
    basename: str = "fig_group_shape_sensitivity",
    lrt_table: Optional[pd.DataFrame] = None,
) -> None:
    """Plot group-specific Weibull shape estimates with bootstrap intervals."""
    tab = table.sort_values("delta_T_C").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.75))
    x = np.arange(len(tab), dtype=float)
    k_hat = tab["k_hat"].to_numpy(dtype=float)
    k_lo = tab["k_ci_lower"].to_numpy(dtype=float)
    k_hi = tab["k_ci_upper"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        k_hat,
        yerr=[k_hat - k_lo, k_hi - k_hat],
        fmt="o",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        capsize=3,
        lw=1.3,
        label="Group-specific k",
    )
    if "k_hat_bias_corrected" in tab.columns:
        ax.scatter(
            x + 0.10,
            tab["k_hat_bias_corrected"].to_numpy(dtype=float),
            s=42,
            facecolors="white",
            edgecolors=PLOT_COLORS["posterior_density"],
            linewidths=1.5,
            marker="o",
            label="Bias-corrected k",
            zorder=4,
        )
    pooled = float(tab["pooled_picm_l_k"].iloc[0])
    ax.axhline(pooled, color=PLOT_COLORS["mle"], lw=1.1, ls="--", label="Pooled PICM-L k")
    if lrt_table is not None and not lrt_table.empty:
        p_value = float(lrt_table["p_value"].iloc[0])
        ax.text(
            0.04,
            0.95,
            f"Common-k LRT p = {p_value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#b5b5b5", alpha=0.94),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}" for d in tab["delta_T_C"]])
    ax.set_xlabel("DeltaT (C)")
    ax.set_ylabel("Weibull shape k")
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=2,
        fontsize=8.0,
        rect_top=0.88,
        legend_y=0.985,
        columnspacing=1.2,
        handlelength=1.8,
    )
    save_simple_figure(fig, out_dir, basename)
    plt.close(fig)


def build_100c_scatter_tables(data: ReliabilityData, reference_max_hours: float = 2678.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build 100 C unit audit and within-group scatter diagnostic tables."""
    grp = data.df[np.isclose(data.df["delta_T"].to_numpy(dtype=float), 100.0)].copy()
    observed = grp.loc[grp["event"].astype(int) == 1].copy()
    unit_table = observed[["unit_id", "ttf_hours", "cycles", "event"]].copy()
    unit_table = unit_table.sort_values("ttf_hours").reset_index(drop=True)

    fail_cycles = observed["cycles"].to_numpy(dtype=float)
    k_hat, _, eta_hat = stats.weibull_min.fit(fail_cycles, floc=0)
    p025, p05, p50, p95, p975 = [
        float(stats.weibull_min.ppf(q, k_hat, loc=0, scale=eta_hat) * CYCLE_TIME_HOURS)
        for q in (0.025, 0.05, 0.50, 0.95, 0.975)
    ]
    obs_min = float(observed["ttf_hours"].min())
    obs_max = float(observed["ttf_hours"].max())
    reconciliation = (
        f"An earlier reference value was 2678 h; the unit-level CSV used for analysis gives {obs_max:.0f} h. "
        "The analysis reports the CSV-verified range."
    )
    scatter_note = SCATTER_100C_NOTE
    scatter_table = pd.DataFrame(
        [
            {
                "delta_T_C": 100.0,
                "n_failed": int(len(observed)),
                "observed_min_hours": obs_min,
                "observed_max_hours": obs_max,
                "reference_max_hours": float(reference_max_hours),
                "csv_verified_max_hours": obs_max,
                "range_reconciliation_note": reconciliation,
                "weibull_k_hat": float(k_hat),
                "weibull_eta_hat_cycles": float(eta_hat),
                "fit_p05_hours": p05,
                "fit_p50_hours": p50,
                "fit_p95_hours": p95,
                "fit_p025_hours": p025,
                "fit_p975_hours": p975,
                "observed_within_p05_p95": bool(obs_min >= p05 and obs_max <= p95),
                "observed_within_p025_p975": bool(obs_min >= p025 and obs_max <= p975),
                "interpretation_note": scatter_note,
            }
        ]
    )
    return unit_table, scatter_table


def compute_failure_only_regression_metrics(
    data: ReliabilityData,
    theta_picm_l: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute descriptive failure-only RMSE and pseudo-R2 metrics."""
    fail = data.df.loc[data.df["event"].astype(int) == 1].copy()
    k, beta1, beta0 = [float(v) for v in theta_picm_l]

    def _predicted_log_median(delta: np.ndarray) -> np.ndarray:
        eta = np.exp(beta0 - beta1 * np.log(delta.astype(float)))
        med = eta * (math.log(2.0) ** (1.0 / k))
        return np.log(med)

    observed = np.log(fail["cycles"].to_numpy(dtype=float))
    predicted = _predicted_log_median(fail["delta_T"].to_numpy(dtype=float))
    residual = observed - predicted
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    rmse = math.sqrt(float(np.mean(residual ** 2)))
    overall = pd.DataFrame(
        [
            {
                "metric_scope": "all_observed_failures",
                "n_points": int(len(fail)),
                "rmse_log_cycles": rmse,
                "multiplicative_rmse": math.exp(rmse),
                "pseudo_R2": 1.0 - sse / sst if sst > 0 else np.nan,
                "warning": FAILURE_ONLY_WARNING,
            }
        ]
    )

    by_rows: List[Dict[str, object]] = []
    for d, grp in fail.groupby("delta_T"):
        obs_g = np.log(grp["cycles"].to_numpy(dtype=float))
        pred_g = _predicted_log_median(np.repeat(float(d), len(grp)))
        res_g = obs_g - pred_g
        sse_g = float(np.sum(res_g ** 2))
        sst_g = float(np.sum((obs_g - np.mean(obs_g)) ** 2))
        rmse_g = math.sqrt(float(np.mean(res_g ** 2)))
        by_rows.append(
            {
                "delta_T_C": float(d),
                "metric_scope": "stress_observed_failures",
                "n_points": int(len(grp)),
                "rmse_log_cycles": rmse_g,
                "multiplicative_rmse": math.exp(rmse_g),
                "pseudo_R2": np.nan,
                "warning": (
                    FAILURE_ONLY_WARNING
                    + " Stress-wise pseudo_R2 is not reported because the fitted "
                    "stress-life predictor is constant within each stress group."
                ),
            }
        )
    by_stress = pd.DataFrame(by_rows).sort_values("delta_T_C").reset_index(drop=True)
    return overall, by_stress


def build_high_deltaT_extrapolation(
    theta_picm_l: np.ndarray,
    targets: np.ndarray = HIGH_DT_TARGETS,
) -> pd.DataFrame:
    """Build MLE high-DeltaT model-conditional extrapolation table."""
    k, beta1, beta0 = [float(v) for v in theta_picm_l]
    rows: List[Dict[str, object]] = []
    for d in _validate_delta_t_values(targets):
        eta = math.exp(beta0 - beta1 * math.log(float(d)))
        b10 = float(weibull_b_quantile(np.array([eta]), np.array([k]), p_fail=0.1)[0])
        med = float(weibull_median(np.array([eta]), np.array([k]))[0])
        rows.append(
            {
                "delta_T_C": float(d),
                "B10_cycles": b10,
                "B10_hours": b10 * CYCLE_TIME_HOURS,
                "median_cycles": med,
                "median_hours": med * CYCLE_TIME_HOURS,
                "evidential_regime": "extrapolative model-conditional",
                "warning": HIGH_DT_WARNING,
            }
        )
    return pd.DataFrame(rows)


def build_high_deltaT_posterior_extrapolation(
    posterior_samples: np.ndarray,
    targets: np.ndarray = HIGH_DT_TARGETS,
) -> pd.DataFrame:
    """Build posterior high-DeltaT model-conditional extrapolation table."""
    targets = _validate_delta_t_values(targets)
    pred = predict_picm_l_quantities(np.asarray(posterior_samples, dtype=float), targets)
    b10_summary = pred["b10_summary"].set_index("delta_T")
    median_summary = pred["median_summary"].set_index("delta_T")
    rows: List[Dict[str, object]] = []
    for d in targets:
        b10 = b10_summary.loc[float(d)]
        med = median_summary.loc[float(d)]
        rows.append(
            {
                "delta_T_C": float(d),
                "B10_q025_cycles": float(b10["b10_q025"]),
                "B10_median_cycles": float(b10["b10_q50"]),
                "B10_q975_cycles": float(b10["b10_q975"]),
                "B10_q025_hours": float(b10["b10_q025"]) * CYCLE_TIME_HOURS,
                "B10_median_hours": float(b10["b10_q50"]) * CYCLE_TIME_HOURS,
                "B10_q975_hours": float(b10["b10_q975"]) * CYCLE_TIME_HOURS,
                "medianlife_q025_cycles": float(med["medianlife_q025"]),
                "medianlife_median_cycles": float(med["medianlife_q50"]),
                "medianlife_q975_cycles": float(med["medianlife_q975"]),
                "evidential_regime": "extrapolative model-conditional",
                "interval_basis": "PICM-L Bayesian posterior equal-tailed interval",
                "warning": HIGH_DT_WARNING,
            }
        )
    return pd.DataFrame(rows)


def build_low_stress_duration(
    posterior_samples: np.ndarray,
    targets: np.ndarray = LOW_STRESS_DURATION_TARGETS,
) -> pd.DataFrame:
    """Build posterior-median low-stress B10 duration equivalence table."""
    targets = _validate_delta_t_values(targets)
    pred = predict_picm_l_quantities(np.asarray(posterior_samples, dtype=float), targets)
    summary = pred["b10_summary"].set_index("delta_T")
    rows: List[Dict[str, object]] = []
    for d in targets:
        b10 = float(summary.loc[float(d), "b10_q50"])
        hours = b10 * CYCLE_TIME_HOURS
        rows.append(
            {
                "delta_T_C": float(d),
                "basis": "posterior_median_B10",
                "B10_cycles": b10,
                "B10_hours": hours,
                "B10_years": hours / (24.0 * 365.0),
                "interpretation_note": LOW_STRESS_DURATION_NOTE,
            }
        )
    return pd.DataFrame(rows)


def build_refreshed_model_comparison_table(
    picm_l_mle: Dict[str, object],
    picm_c: Dict[str, object],
    lognorm: Dict[str, object],
    bayes_ic: Dict[str, float],
    n_obs: int,
) -> pd.DataFrame:
    """Build model-comparison summary table."""
    ll_l = float(picm_l_mle["loglik"])
    aic_l, bic_l = calc_aic_bic(ll_l, n_params=3, n_obs=n_obs)
    return pd.DataFrame(
        [
            {
                "model": "Weibull PICM-L",
                "loglik": ll_l,
                "n_parameters": 3,
                "AIC": aic_l,
                "BIC": bic_l,
                "WAIC": float(bayes_ic["waic"]),
                "DIC": float(bayes_ic["dic"]),
                "interpretation_note": MODEL_COMPARISON_NOTE,
            },
            {
                "model": "Weibull PICM-C",
                "loglik": float(picm_c["loglik"]),
                "n_parameters": 4,
                "AIC": float(picm_c["aic"]),
                "BIC": float(picm_c["bic"]),
                "WAIC": np.nan,
                "DIC": np.nan,
                "interpretation_note": MODEL_COMPARISON_NOTE,
            },
            {
                "model": "Lognormal AFT",
                "loglik": float(lognorm["loglik"]),
                "n_parameters": 3,
                "AIC": float(lognorm["aic"]),
                "BIC": float(lognorm["bic"]),
                "WAIC": np.nan,
                "DIC": np.nan,
                "interpretation_note": MODEL_COMPARISON_NOTE,
            },
        ]
    )


def _find_probability_crossing(delta_grid: np.ndarray, probability: np.ndarray, level: float) -> float:
    """Linearly interpolate the DeltaT crossing where probability equals level."""
    d = np.asarray(delta_grid, dtype=float)
    p = np.asarray(probability, dtype=float)
    order = np.argsort(d)
    d = d[order]
    p = p[order]
    diff = p - float(level)
    exact = np.where(np.isclose(diff, 0.0, atol=1e-12))[0]
    if exact.size:
        return float(d[int(exact[0])])
    for i in range(len(d) - 1):
        if diff[i] == 0:
            return float(d[i])
        if diff[i] * diff[i + 1] < 0:
            if np.isclose(p[i + 1], p[i]):
                return float(d[i])
            return float(d[i] + (float(level) - p[i]) * (d[i + 1] - d[i]) / (p[i + 1] - p[i]))
    return float("nan")


def build_decision_boundary_uncertainty(
    posterior_samples: np.ndarray,
    delta_grid: np.ndarray = DECISION_PROB_DELTA_GRID,
    mission_targets: Sequence[float] = MISSION_CYCLE_TARGETS,
    probability_levels: np.ndarray = DECISION_PROBABILITY_LEVELS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pr[B10(DeltaT) >= N] and crossing summaries from posterior samples."""
    grid = _validate_delta_t_values(delta_grid)
    pred = predict_picm_l_quantities(np.asarray(posterior_samples, dtype=float), grid)
    b10 = np.asarray(pred["b10_samples"], dtype=float)

    prob_rows: List[Dict[str, object]] = []
    crossing_rows: List[Dict[str, object]] = []
    for mission in [float(m) for m in mission_targets]:
        probs = np.mean(b10 >= mission, axis=0)
        for d, p in zip(grid, probs):
            prob_rows.append(
                {
                    "delta_T_C": float(d),
                    "mission_cycles": mission,
                    "posterior_probability_B10_exceeds_mission": float(p),
                }
            )
        for level in np.asarray(probability_levels, dtype=float):
            crossing = _find_probability_crossing(grid, probs, float(level))
            crossing_rows.append(
                {
                    "mission_cycles": mission,
                    "probability_level": float(level),
                    "delta_T_crossing_C": crossing,
                    "crossing_regime": (
                        classify_evidential_regime(crossing)
                        if np.isfinite(crossing)
                        else "not crossed within evaluated grid"
                    ),
                    "interpretation_note": DECISION_BOUNDARY_NOTE,
                }
            )
    return pd.DataFrame(crossing_rows), pd.DataFrame(prob_rows)


def plot_decision_boundary_uncertainty(
    probability_table: pd.DataFrame,
    out_dir: Path,
    basename: str = "fig_model_conditional_decision_boundary_uncertainty",
) -> None:
    """Plot posterior decision probability Pr(B10 >= N) versus DeltaT."""
    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.95))
    colors = [PLOT_COLORS["bayes_fit"], PLOT_COLORS["posterior_density"], PLOT_COLORS["picm_c"]]
    for idx, (mission, grp) in enumerate(probability_table.groupby("mission_cycles")):
        g = grp.sort_values("delta_T_C")
        ax.plot(
            g["delta_T_C"],
            g["posterior_probability_B10_exceeds_mission"],
            lw=1.8,
            color=colors[idx % len(colors)],
            label=f"N = {int(mission)} cycles",
        )
    for level in (0.90, 0.95):
        ax.axhline(level, color=PLOT_COLORS["aux"], lw=0.85, ls=":", alpha=0.85)
        ax.text(
            float(probability_table["delta_T_C"].min()) + 0.5,
            level + 0.01,
            f"{level:.2f}",
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=PLOT_COLORS["aux"],
        )
    ax.axvline(OBSERVED_DOMAIN_MIN, color=PLOT_COLORS["mle"], lw=0.9, ls="--", alpha=0.75)
    ax.set_xlabel("DeltaT (C)")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Model-conditional probability that B10 >= N")
    style_axis(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    save_simple_figure(fig, out_dir, basename)
    plt.close(fig)


def build_acceptance_checks_v3p1(
    mle_theta: np.ndarray,
    mle_b10_35: float,
    firth_theta: np.ndarray,
    scatter_100c: pd.DataFrame,
    group_shape: pd.DataFrame,
    group_shape_lrt: pd.DataFrame,
    failure_metrics: pd.DataFrame,
    high_dt: pd.DataFrame,
    accept_rates: np.ndarray,
    rhat: np.ndarray,
    ess: np.ndarray,
    b10_bayes_35: float,
    low_duration: pd.DataFrame,
) -> pd.DataFrame:
    """Build fail-loud deterministic acceptance checks for V5 outputs."""
    checks: List[Dict[str, object]] = []

    def add_numeric_check(name: str, value: float, expected: float, tolerance: float, detail: str = "") -> None:
        checks.append(
            {
                "check_name": name,
                "value": float(value),
                "expected": float(expected),
                "tolerance": float(tolerance),
                "passed": bool(abs(float(value) - float(expected)) <= float(tolerance)),
                "detail": detail,
            }
        )

    def add_boolean_check(name: str, passed: bool, value: object, detail: str = "") -> None:
        checks.append(
            {
                "check_name": name,
                "value": value,
                "expected": True,
                "tolerance": "",
                "passed": bool(passed),
                "detail": detail,
            }
        )

    add_numeric_check("MLE_k", float(mle_theta[0]), 2.98824797, 1e-3)
    add_numeric_check("MLE_beta1", float(mle_theta[1]), 2.43584622, 1e-3)
    add_numeric_check("MLE_beta0", float(mle_theta[2]), 17.66195450, 1e-3)
    add_numeric_check("MLE_B10_35C", float(mle_b10_35), 3822.3, 2.0)
    add_numeric_check("CSV_100C_max_hours", float(scatter_100c["csv_verified_max_hours"].iloc[0]), 2818.0, 0.1)
    add_numeric_check("failure_only_RMSE_log_cycles", float(failure_metrics["rmse_log_cycles"].iloc[0]), 0.404, 0.01)
    add_numeric_check("failure_only_pseudo_R2", float(failure_metrics["pseudo_R2"].iloc[0]), 0.534, 0.01)

    for d, expected in [(100.0, 2.79), (125.0, 3.87), (150.0, 4.32)]:
        row = group_shape.loc[np.isclose(group_shape["delta_T_C"], d)].iloc[0]
        add_numeric_check(f"group_k_{int(d)}C", float(row["k_hat"]), expected, 0.05)
    for d, expected in [(175.0, 75.8), (200.0, 54.8)]:
        row = high_dt.loc[np.isclose(high_dt["delta_T_C"], d)].iloc[0]
        add_numeric_check(f"high_deltaT_B10_{int(d)}C", float(row["B10_cycles"]), expected, 0.5)

    firth_k_diff = abs(float(firth_theta[0]) - float(mle_theta[0]))
    add_numeric_check("firth_agrees_with_mle", firth_k_diff, 0.0, 0.05, "|firth_k - mle_k|")

    scatter_ok = bool(scatter_100c["observed_within_p025_p975"].iloc[0])
    add_boolean_check("scatter_within_p025_p975", scatter_ok, scatter_ok, "Headline 100 C containment uses no tolerance.")

    lrt_p = float(group_shape_lrt["p_value"].iloc[0])
    add_boolean_check(
        "group_shape_lrt_pvalue_finite",
        bool(np.isfinite(lrt_p) and (0.0 <= lrt_p <= 1.0)),
        lrt_p,
        "Common-k likelihood-ratio p-value must be finite and bounded.",
    )

    acc = np.asarray(accept_rates, dtype=float)
    add_boolean_check(
        "mcmc_all_chains_in_band",
        bool(np.all((acc >= MCMC_ACCEPT_LOWER) & (acc <= MCMC_ACCEPT_UPPER))),
        ", ".join(f"{a:.6f}" for a in acc),
        f"Post-adaptation sampling acceptance in [{MCMC_ACCEPT_LOWER:.2f}, {MCMC_ACCEPT_UPPER:.2f}].",
    )
    add_boolean_check("mcmc_rhat_max_lt_1p01", bool(float(np.max(rhat)) < 1.01), float(np.max(rhat)))
    add_boolean_check("mcmc_ess_min_gt_2000", bool(float(np.min(ess)) > 2000.0), float(np.min(ess)))
    add_numeric_check("bayes_B10_35C_median", float(b10_bayes_35), 3842.0, 60.0)

    duration_by_stress = low_duration.set_index("delta_T_C")
    add_numeric_check("duration_25C_years", float(duration_by_stress.loc[25.0, "B10_years"]), 3.0, 0.6)
    add_numeric_check("duration_20C_years", float(duration_by_stress.loc[20.0, "B10_years"]), 5.2, 0.8)
    return pd.DataFrame(checks)


def print_and_assert_acceptance_checks(checks: pd.DataFrame) -> None:
    """Print the full V5 validation table and stop on any failed check."""
    print("\n================= V5 Validation Checks =================")
    print(checks.to_string(index=False))
    print("==========================================================\n")
    if not bool(np.all(checks["passed"].to_numpy(dtype=bool))):
        failing = checks.loc[~checks["passed"].astype(bool)].copy()
        print("Failing V5 validation checks:")
        print(failing.to_string(index=False))
        raise SystemExit(1)


def build_mcmc_diagnostics_table(
    accept_rates: np.ndarray,
    rhat: np.ndarray,
    ess: np.ndarray,
) -> pd.DataFrame:
    """Create machine-readable MCMC diagnostics table."""
    rows = []
    for c, acc in enumerate(accept_rates, start=1):
        rows.append(
            {
                "diagnostic_type": "chain_acceptance",
                "name": f"chain_{c}",
                "value": float(acc),
                "basis": "post_adaptation_sampling_phase",
                "warn": bool((acc < MCMC_ACCEPT_LOWER) or (acc > MCMC_ACCEPT_UPPER)),
            }
        )
    param_names = ["k", "beta1", "beta0"]
    for i, p in enumerate(param_names):
        rows.append(
            {
                "diagnostic_type": "rhat",
                "name": p,
                "value": float(rhat[i]),
                "basis": "retained_sampling_phase",
                "warn": bool(rhat[i] > 1.01),
            }
        )
        rows.append(
            {
                "diagnostic_type": "ess",
                "name": p,
                "value": float(ess[i]),
                "basis": "retained_sampling_phase",
                "warn": bool(ess[i] < 2000.0),
            }
        )
    return pd.DataFrame(rows)


def write_analysis_results_summary(
    out_file: Path,
    modelwise_table: pd.DataFrame,
    envelope_table: pd.DataFrame,
    mission_table: pd.DataFrame,
) -> str:
    """Build and save a concise narrative block for results summary."""
    env = envelope_table.set_index("delta_T")
    lines = [
        "Low-DeltaT Results Narrative",
        "============================",
        f"Observed domain directly supported by data: DeltaT in [{OBSERVED_DOMAIN_MIN:.0f}, {OBSERVED_DOMAIN_MAX:.0f}] C.",
        f"Extrapolation domain: DeltaT < {EXTRAPOLATION_CUTOFF:.0f} C (not directly validated by observed failures).",
        "At 25C and 50C all units are right-censored, so 10-20C life estimates are extrapolative and model-sensitive.",
        "",
        "Model-form B10 envelope (cycles) at target DeltaT:",
    ]
    for d in TARGET_DELTA_TS:
        r = env.loc[float(d)]
        lines.append(
            f"  DeltaT={int(d)}C: B10 median range [{r['b10_median_min']:.1f}, {r['b10_median_max']:.1f}], "
            f"95% envelope [{r['b10_lower95_min']:.1f}, {r['b10_upper95_max']:.1f}]"
        )
    lines.append("")
    lines.append("Mission survival probabilities by DeltaT (model-averaged envelope across PICM-L/C and lognormal):")
    for mission in sorted(mission_table["mission_cycles"].unique()):
        sub_m = mission_table[mission_table["mission_cycles"] == mission]
        lines.append(f"  Mission {mission:.0f} cycles:")
        for d in TARGET_DELTA_TS:
            s = sub_m[sub_m["delta_T"] == float(d)]
            lo = float(np.min(s["p_survive_q025"]))
            hi = float(np.max(s["p_survive_q975"]))
            md_lo = float(np.min(s["p_survive_q50"]))
            md_hi = float(np.max(s["p_survive_q50"]))
            lines.append(
                f"    DeltaT={int(d)}C: median-survival range [{md_lo:.3f}, {md_hi:.3f}], "
                f"95% envelope [{lo:.3f}, {hi:.3f}]"
            )

    text = "\n".join(lines) + "\n"
    out_file.write_text(text, encoding="utf-8")
    return text


def compute_failure_fraction_crosscheck(
    data: ReliabilityData,
    bayes_samples: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """
    Cross-check observed stress-wise failure fractions against posterior-predictive fractions.

    The predictive fraction is computed for each stress at its observed runout, using n=8 Bernoulli
    outcomes per stress and summarizing the posterior-predictive distribution.
    """
    rng = np.random.default_rng(seed)
    k_s = bayes_samples[:, 0]
    b1_s = bayes_samples[:, 1]
    b0_s = bayes_samples[:, 2]

    rows = []
    for s in data.stress_levels:
        idx = data.stress_to_idx[float(s)]
        dsub = data.df.iloc[idx]
        ev = dsub["event"].to_numpy(dtype=int)
        n_total = int(ev.size)
        n_failed = int(np.sum(ev))
        obs_frac = float(np.mean(ev))

        cens = dsub[dsub["event"] == 0]["cycles"].to_numpy(dtype=float)
        runout = float(np.max(cens)) if cens.size > 0 else float(np.max(dsub["cycles"].to_numpy(dtype=float)))

        x = np.log(s)
        eta = np.exp(b0_s - b1_s * x)
        p_fail = 1.0 - np.exp(-((runout / eta) ** k_s))
        pred_counts = rng.binomial(n_total, np.clip(p_fail, 1e-12, 1.0 - 1e-12))
        frac = pred_counts / max(n_total, 1)

        ql, qm, qu = np.quantile(frac, [0.025, 0.5, 0.975])
        rows.append(
            {
                "delta_T": int(s),
                "observed_failed": n_failed,
                "observed_total": n_total,
                "observed_failure_fraction": obs_frac,
                "predicted_median_failure_fraction": float(qm),
                "predicted_pi_2.5%": float(ql),
                "predicted_pi_97.5%": float(qu),
                "observed_within_95ppi": bool((obs_frac >= ql) and (obs_frac <= qu)),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_T").reset_index(drop=True)


def run_parameterization_consistency_checks(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    bayes_samples: np.ndarray,
) -> pd.DataFrame:
    """Run lightweight internal checks for reliability parameterization consistency."""
    rows: List[Dict[str, object]] = []

    def _append(name: str, passed: bool, value: float, threshold: float, detail: str) -> None:
        rows.append(
            {
                "check_name": name,
                "passed": bool(passed),
                "value": float(value),
                "threshold": float(threshold),
                "detail": detail,
            }
        )

    # Check 1: B10 closed form at use condition.
    th_med = np.median(bayes_samples, axis=0)
    max_diff_b10 = 0.0
    for th in [np.asarray(mle_theta, dtype=float), np.asarray(th_med, dtype=float)]:
        k, b1, b0 = [float(v) for v in th]
        eta_use = float(np.exp(b0 - b1 * np.log(DELTA_T_USE)))
        b10_a = float(b10_use_from_theta(th, DELTA_T_USE))
        b10_b = float(weibull_b_quantile(eta_use, k, p_fail=0.1))
        max_diff_b10 = max(max_diff_b10, abs(b10_a - b10_b))
    _append(
        "B10_formula_consistency",
        passed=max_diff_b10 <= 1e-9,
        value=max_diff_b10,
        threshold=1e-9,
        detail="b10_use_from_theta equals Weibull quantile form at p=0.1.",
    )

    # Check 2: Weibull characteristic life interpretation S(eta)=exp(-1).
    eta_t, k_t = 1234.5, 2.7
    s_eta = float(np.exp(-((eta_t / eta_t) ** k_t)))
    diff_s_eta = abs(s_eta - np.exp(-1.0))
    _append(
        "Weibull_scale_characteristic_life",
        passed=diff_s_eta <= 1e-12,
        value=diff_s_eta,
        threshold=1e-12,
        detail="Survival at y=eta should equal exp(-1).",
    )

    # Check 3: Weibull median formula.
    med_a = float(weibull_median(eta_t, k_t))
    med_b = float(weibull_b_quantile(eta_t, k_t, p_fail=0.5))
    diff_median = abs(med_a - med_b)
    _append(
        "Weibull_median_consistency",
        passed=diff_median <= 1e-12,
        value=diff_median,
        threshold=1e-12,
        detail="weibull_median equals Weibull p=0.5 quantile.",
    )

    # Check 4: Stress-life monotonicity under posterior draws (PICM-L).
    delta = np.linspace(np.min(B10_GRID_DELTA_T), np.max(B10_GRID_DELTA_T), 120)
    x = np.log(delta)[None, :]
    k_s = bayes_samples[:, 0][:, None]
    b1_s = bayes_samples[:, 1][:, None]
    b0_s = bayes_samples[:, 2][:, None]
    eta_draws = np.exp(b0_s - b1_s * x)
    monotone_mask = np.all(np.diff(eta_draws, axis=1) <= 1e-10, axis=1)
    mono_prop = float(np.mean(monotone_mask))
    _append(
        "Stress_life_monotonicity_posterior",
        passed=mono_prop >= 0.95,
        value=mono_prop,
        threshold=0.95,
        detail="Proportion of posterior draws with non-increasing life as DeltaT increases.",
    )

    # Check 5: Exact censoring contribution form in likelihood.
    ll_i = np.asarray(loglik_weibull_picm_l(np.asarray(mle_theta, dtype=float), data, pointwise=True), dtype=float)
    k_m, b1_m, b0_m = [float(v) for v in mle_theta]
    fail_idx = np.where(data.event == 1)[0]
    cen_idx = np.where(data.event == 0)[0]
    max_diff_lik = 0.0
    if fail_idx.size > 0:
        i = int(fail_idx[0])
        y = float(data.y[i])
        log_eta = float(b0_m - b1_m * data.x_log[i])
        z = float((y / np.exp(log_eta)) ** k_m)
        ll_manual = float(np.log(k_m) + (k_m - 1.0) * (np.log(y) - log_eta) - log_eta - z)
        max_diff_lik = max(max_diff_lik, abs(ll_manual - float(ll_i[i])))
    if cen_idx.size > 0:
        i = int(cen_idx[0])
        y = float(data.y[i])
        eta = float(np.exp(b0_m - b1_m * data.x_log[i]))
        ll_manual = float(-((y / eta) ** k_m))
        max_diff_lik = max(max_diff_lik, abs(ll_manual - float(ll_i[i])))
    _append(
        "Censoring_likelihood_contribution",
        passed=max_diff_lik <= 1e-9,
        value=max_diff_lik,
        threshold=1e-9,
        detail="Pointwise failure and right-censor terms match analytic forms.",
    )

    # Check 6: Lognormal AFT median mapping.
    mu_t, sigma_t = 7.3, 0.45
    diff_logn = abs(float(lognormal_b_quantile(mu_t, sigma_t, p_fail=0.5)) - float(lognormal_median(mu_t)))
    _append(
        "Lognormal_AFT_median_mapping",
        passed=diff_logn <= 1e-12,
        value=diff_logn,
        threshold=1e-12,
        detail="lognormal_b_quantile(p=0.5) equals exp(mu).",
    )
    return pd.DataFrame(rows)


def _summarize_interval(x: np.ndarray) -> Tuple[float, float, float]:
    """Return median and 95% equal-tailed interval."""
    arr = np.asarray(x, dtype=float)
    return float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def compute_loso_predictive_adequacy_picm_l(
    data: ReliabilityData,
    prior_cfg: Optional[Dict[str, float]] = None,
    max_stress: Optional[int] = None,
) -> pd.DataFrame:
    """Leave-one-stress-out predictive adequacy table for PICM-L using train-only posterior predictive draws."""
    cfg = get_prior_config(BASELINE_PRIOR_KEY) if prior_cfg is None else prior_cfg
    stress_all = np.array(sorted(data.stress_levels), dtype=float)
    if max_stress is not None:
        stress_all = stress_all[: int(max(1, max_stress))]

    rows: List[Dict[str, object]] = []
    for s in stress_all:
        df_train = data.df[data.df["delta_T"] != s].copy()
        d_train = prepare_data(df_train)
        fit_train = fit_picm_l_mle(d_train, n_starts=12, seed=SEED_OPT + int(s) + 700)

        starts = make_picm_l_mcmc_starts(fit_train["theta"], prior_cfg=cfg)
        for i in range(len(starts)):
            if not np.isfinite(logposterior_picm_l(starts[i], d_train, prior_cfg=cfg)):
                starts[i] = np.array(
                    [
                        cfg["k_trunc_lower"] + 1.0 + 0.1 * i,
                        max(cfg["beta1_mean"], cfg["beta1_trunc_lower"] + 0.2 + 0.1 * i),
                        cfg["beta0_mean"] - 0.4 * i,
                    ],
                    dtype=float,
                )

        seeds = [SEED_MCMC[0] + int(s) + 5000, SEED_MCMC[1] + int(s) + 5000, SEED_MCMC[2] + int(s) + 5000]
        mh = run_adaptive_mh_picm_l(
            data=d_train,
            starts=starts,
            seeds=seeds,
            burnin=LOSO_MCMC_BURNIN,
            keep=LOSO_MCMC_KEEP,
            adapt_block=LOSO_MCMC_ADAPT_BLOCK,
            prior_cfg=cfg,
        )
        draws = mh["chains"].reshape(-1, 3)

        idx = data.stress_to_idx[float(s)]
        d_hold = data.df.iloc[idx]
        n_total = int(d_hold.shape[0])
        n_fail = int(np.sum(d_hold["event"].to_numpy(dtype=int)))
        obs_frac = float(n_fail / max(n_total, 1))
        runout = float(np.max(d_hold["cycles"].to_numpy(dtype=float)))

        eta = np.exp(draws[:, 2] - draws[:, 1] * np.log(s))
        p_fail = 1.0 - np.exp(-((runout / eta) ** draws[:, 0]))
        rng = np.random.default_rng(SEED_PRED + int(s) + 9000)
        pred_counts = rng.binomial(n_total, np.clip(p_fail, 1e-12, 1.0 - 1e-12))
        pred_frac = pred_counts / max(n_total, 1)
        pred_q50, pred_q025, pred_q975 = _summarize_interval(pred_frac)

        rows.append(
            {
                "model": "PICM-L",
                "uncertainty_method": "MCMC posterior predictive",
                "held_out_stress": float(s),
                "observed_failed": n_fail,
                "observed_total": n_total,
                "observed_failure_fraction": obs_frac,
                "predicted_failure_fraction": pred_q50,
                "predicted_pi_2_5": pred_q025,
                "predicted_pi_97_5": pred_q975,
                "absolute_error": abs(pred_q50 - obs_frac),
                "coverage_95ppi": bool((obs_frac >= pred_q025) and (obs_frac <= pred_q975)),
            }
        )
    return pd.DataFrame(rows).sort_values("held_out_stress").reset_index(drop=True)


def run_prior_sensitivity_picm_l(
    data: ReliabilityData,
    prior_keys: Sequence[str],
    mission_cycles: Sequence[float],
) -> Dict[str, object]:
    """
    Prior sensitivity workflow for PICM-L.

    Outputs a compact summary for:
    - B10 at 20C, 25C, and 35C
    - P(T > mission_cycles) at 20C, 25C, and 35C
    """
    delta_targets = PRIOR_SENS_DELTA_TARGETS.copy()
    rows: List[Dict[str, object]] = []
    draws_by_prior: Dict[str, np.ndarray] = {}
    b10_35_by_prior: Dict[str, np.ndarray] = {}

    for i, key in enumerate(prior_keys):
        cfg = get_prior_config(key)
        mle_i = fit_picm_l_mle(data, n_starts=max(16, N_OPT_STARTS_MAIN // 2), seed=SEED_OPT + 100 + i)
        starts = make_picm_l_mcmc_starts(mle_i["theta"], prior_cfg=cfg)
        for j in range(len(starts)):
            if not np.isfinite(logposterior_picm_l(starts[j], data, prior_cfg=cfg)):
                starts[j] = np.array(
                    [
                        cfg["k_trunc_lower"] + 1.0 + 0.1 * j,
                        max(cfg["beta1_mean"], cfg["beta1_trunc_lower"] + 0.2 + 0.1 * j),
                        cfg["beta0_mean"] - 0.4 * j,
                    ],
                    dtype=float,
                )

        seeds = [SEED_MCMC[0] + 300 * (i + 1), SEED_MCMC[1] + 300 * (i + 1), SEED_MCMC[2] + 300 * (i + 1)]
        mh_i = run_adaptive_mh_picm_l(
            data=data,
            starts=starts,
            seeds=seeds,
            burnin=PRIOR_SENS_BURNIN,
            keep=PRIOR_SENS_KEEP,
            adapt_block=MCMC_ADAPT_BLOCK,
            prior_cfg=cfg,
        )
        draws_i = mh_i["chains"].reshape(-1, 3)
        if draws_i.shape[0] > PRIOR_SENS_DRAW_SUBSAMPLE:
            rng_sub = np.random.default_rng(SEED_PRED + 20000 + i)
            idx = rng_sub.choice(draws_i.shape[0], size=PRIOR_SENS_DRAW_SUBSAMPLE, replace=False)
            draws_i = draws_i[idx, :]
        draws_by_prior[key] = draws_i

        pred_i = predict_picm_l_quantities(draws_i, delta_targets)
        b10_i = pred_i["b10_samples"]
        for j, d in enumerate(delta_targets):
            med, q025, q975 = _summarize_interval(b10_i[:, j])
            rows.append(
                {
                    "prior_setting": key,
                    "quantity": "B10_cycles",
                    "delta_T": float(d),
                    "mission_cycles": np.nan,
                    "median": med,
                    "q025": q025,
                    "q975": q975,
                    "uncertainty_method": "MCMC posterior",
                }
            )
            if abs(d - 35.0) < 1e-9:
                b10_35_by_prior[key] = np.asarray(b10_i[:, j], dtype=float)

        for d in delta_targets:
            for mission in mission_cycles:
                surv = prob_survive_mission(draws_i, np.array([d], dtype=float), mission, model_name="picm_l")["survival_samples"][:, 0]
                med, q025, q975 = _summarize_interval(surv)
                rows.append(
                    {
                        "prior_setting": key,
                        "quantity": "Survival_probability",
                        "delta_T": float(d),
                        "mission_cycles": float(mission),
                        "median": med,
                        "q025": q025,
                        "q975": q975,
                        "uncertainty_method": "MCMC posterior",
                    }
                )
    return {
        "summary_table": pd.DataFrame(rows),
        "draws_by_prior": draws_by_prior,
        "b10_35_by_prior": b10_35_by_prior,
    }


# =============================================================================
# Plotting utilities
# =============================================================================


def set_plot_style() -> None:
    """Journal-style restrained plotting defaults."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (IEEE_ONE_COL_WIDE, 2.8),
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "font.weight": "semibold",
            "axes.labelsize": 9.0,
            "axes.titlesize": 8.9,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlepad": 4.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.grid": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3742",
            "axes.labelcolor": "#2f3742",
            "xtick.color": "#2f3742",
            "ytick.color": "#2f3742",
            "text.color": "#2f3742",
            "axes.axisbelow": True,
            "grid.alpha": 0.10,
            "grid.color": "#9ca3af",
            "grid.linestyle": "-",
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "legend.borderpad": 0.15,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.9,
            "legend.handletextpad": 0.5,
            "lines.linewidth": 1.7,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "mathtext.fontset": "dejavusans",
        }
    )


def setup_plot_style() -> None:
    """Public entry point for centralized plotting configuration."""
    set_plot_style()


def style_axis(ax: plt.Axes) -> None:
    """Apply consistent publication-style axis cosmetics."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color(PLOT_COLORS["aux"])
    ax.spines["bottom"].set_color(PLOT_COLORS["aux"])
    ax.tick_params(axis="both", which="major", length=3.8, width=0.85, colors=PLOT_COLORS["aux"])
    ax.tick_params(axis="both", which="minor", length=2.1, width=0.7, colors=PLOT_COLORS["aux"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")
    ax.grid(True, which="major")


def dedup_legend(handles: Sequence[object], labels: Sequence[str]) -> Tuple[List[object], List[str]]:
    """De-duplicate legend entries while preserving order."""
    out_h, out_l = [], []
    for h, l in zip(handles, labels):
        if l not in out_l:
            out_h.append(h)
            out_l.append(l)
    return out_h, out_l


def add_figure_legend_above(
    fig: plt.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    ncol: int,
    fontsize: float = 7.0,
    rect_top: float = 0.91,
    legend_y: float = 0.982,
    columnspacing: float = 0.9,
    handlelength: float = 1.8,
) -> Optional[object]:
    """Place a figure-level legend above the axes and reserve matching headroom."""
    if not handles:
        return None
    fig.tight_layout(rect=[0.035, 0.035, 0.985, rect_top], pad=0.45)
    return fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        columnspacing=columnspacing,
        handlelength=handlelength,
        borderaxespad=0.2,
    )


def normalize_mojibake_text(text: str) -> str:
    """Return label text after the source-level ASCII normalization step."""
    return text


def normalize_figure_text(fig: plt.Figure) -> None:
    """Normalize titles, labels, legends, and annotations before saving."""
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text(normalize_mojibake_text(suptitle.get_text()))
        suptitle.set_fontweight("bold")

    legends = list(getattr(fig, "legends", []))
    for ax in fig.axes:
        ax.set_title(normalize_mojibake_text(ax.get_title()))
        ax.set_xlabel(normalize_mojibake_text(ax.get_xlabel()))
        ax.set_ylabel(normalize_mojibake_text(ax.get_ylabel()))
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("semibold")
        for text_artist in ax.texts:
            text_artist.set_text(normalize_mojibake_text(text_artist.get_text()))
            text_artist.set_fontweight("semibold")
        legend = ax.get_legend()
        if legend is not None:
            legends.append(legend)

    for legend in legends:
        for text_artist in legend.get_texts():
            text_artist.set_text(normalize_mojibake_text(text_artist.get_text()))
            text_artist.set_fontweight("semibold")


def _figure_group_from_dir(out_dir: Path) -> str:
    """Map a requested output directory to the manifest figure group."""
    try:
        resolved = Path(out_dir).resolve()
        mapping = {
            MAIN_FIG_DIR.resolve(): "main",
            SUPP_FIG_DIR.resolve(): "supplementary",
        }
        return mapping.get(resolved, "supplementary")
    except OSError:
        return "supplementary"


def _infer_columns_from_figure(fig: plt.Figure) -> str:
    """Infer single- or double-column output from the caller's current figure width."""
    width = float(fig.get_size_inches()[0])
    threshold = 0.5 * (IEEE_ONE_COL_WIDE + IEEE_TWO_COL_WIDE)
    return "double" if width >= threshold else "single"


def _title_snapshot(fig: plt.Figure) -> str:
    """Capture suptitle and axes titles before publication stripping."""
    titles: List[str] = []
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        titles.append(normalize_mojibake_text(suptitle.get_text().strip()))
    for ax in fig.axes:
        t = ax.get_title().strip()
        if t:
            titles.append(normalize_mojibake_text(t))
    return " | ".join(dict.fromkeys(titles))


def save_pub_figure(
    fig: plt.Figure,
    name: str,
    group: str,
    *,
    columns: str = "single",
    publication: bool = True,
) -> Dict[str, Path]:
    """Single canonical figure writer. Emits a 600-dpi PNG into figures/all/."""
    if "/" in name or "\\" in name or "." in Path(name).name:
        raise ValueError("Figure name must be a stem only, with no path or extension.")
    if group not in {"main", "supplementary"}:
        raise ValueError(f"Unknown figure group: {group}")
    if columns not in {"single", "double"}:
        raise ValueError(f"Unknown figure column setting: {columns}")

    out_dir = ALL_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    width = IEEE_ONE_COL_WIDE if columns == "single" else IEEE_TWO_COL_WIDE
    current = fig.get_size_inches()
    fig.set_size_inches(width, float(current[1]))
    normalize_figure_text(fig)
    title_original = _title_snapshot(fig)

    if publication:
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            suptitle.set_text("")
        if len(fig.axes) <= 1:
            for ax in fig.axes:
                ax.set_title("")

    fig.canvas.draw()
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)

    existing_idx = next((i for i, row in enumerate(PUB_FIGURE_MANIFEST) if row["name"] == name), None)
    row = {
        "name": name,
        "group": group,
        "columns": columns,
        "png_path": png_path,
        "title_original": title_original,
    }
    if existing_idx is None:
        PUB_FIGURE_MANIFEST.append(row)
    else:
        PUB_FIGURE_MANIFEST[existing_idx] = row
    return {"png": png_path}


def save_figure_dual(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    manifest: List[Dict[str, str]],
    category: str,
    purpose: str,
) -> None:
    """Save a figure and register it in the run manifest."""
    group = _figure_group_from_dir(out_dir)
    paths = save_pub_figure(fig, stem, group, columns=_infer_columns_from_figure(fig))
    manifest.append({"filename": str(paths["png"].relative_to(OUT_DIR)), "category": category, "purpose": purpose})



def _format_delta_t_label(value: float) -> str:
    """Compact Delta T label for figure annotations."""
    return f"Delta T = {int(round(float(value)))} deg C"


def _draw_reference_markers(
    ax: plt.Axes,
    values: Sequence[float],
    y_text: Optional[float] = None,
    annotate: bool = True,
) -> None:
    """Draw thin vertical guides for interpretive reference stresses."""
    if y_text is None:
        y0, y1 = ax.get_ylim()
        if ax.get_yscale() == 'log':
            y_text = 10 ** (0.96 * np.log10(y1) + 0.04 * np.log10(max(y0, 1e-9)))
        else:
            y_text = y0 + 0.93 * (y1 - y0)
    labels = {10.0: '10 deg C', 15.0: '15 deg C', 25.0: '25 deg C', 35.0: '35 deg C'}
    for val in values:
        if float(val) in (10.0, 15.0):
            ls = ':'
            alpha = 0.65
        elif float(val) == 25.0:
            ls = '--'
            alpha = 0.8
        else:
            ls = '-.'
            alpha = 0.9
        ax.axvline(float(val), color='#6b7280', lw=0.85, ls=ls, alpha=min(alpha, 0.65), zorder=0)
        if annotate:
            ax.text(float(val) + 0.6, y_text, labels.get(float(val), f'{float(val):g} deg C'), fontsize=7.2, color='#4b5563', va='top')


def _add_regime_bands(ax: plt.Axes, xmin: float, xmax: float) -> None:
    """Background regime shading used in the revised main figures."""
    ax.axvspan(xmin, EXTRAPOLATION_CUTOFF, color="#fef3c7", alpha=0.16, zorder=0)
    ax.axvspan(EXTRAPOLATION_CUTOFF, LOW_INFO_TESTED_MAX, color="#e5e7eb", alpha=0.10, zorder=0)
    ax.axvspan(LOWEST_FAILURE_OBSERVED_STRESS, xmax, color="#dcfce7", alpha=0.08, zorder=0)


def _add_regime_labels(ax: plt.Axes) -> None:
    """Keep regime naming in the caption to avoid crowding the plot area."""
    return


def write_figure_manifest(manifest: List[Dict[str, str]], out_path: Path) -> None:
    """Write figure manifest for submission assembly."""
    lines = ["Figure Manifest", "================", ""]
    for row in manifest:
        lines.append(f"{row['filename']} - {row['category']} - {row['purpose']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slot_filename(slot: str) -> str:
    """Convert a figure slot such as 'Fig. S1' into a stable PNG filename."""
    return slot.replace(".", "").replace(" ", "") + ".png"


def _build_contact_sheet(manifest_df: pd.DataFrame, out_path: Path) -> None:
    """Build a compact visual contact sheet from saved PNG copies."""
    if manifest_df.empty:
        return
    n = int(manifest_df.shape[0])
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 2.65), dpi=150)
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, (_, row) in zip(axes_arr, manifest_df.iterrows()):
        png_path = OUT_DIR / str(row["png_relpath"])
        img = plt.imread(str(png_path))
        ax.imshow(img)
        title = str(row["slot"]) if str(row["slot"]) else str(row["name"])
        original = str(row.get("title_original", "") or "")
        if original:
            title = f"{title}: {original[:80]}"
        ax.set_title(title, fontsize=7.0, loc="left")
    fig.tight_layout(pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.canvas.print_png(str(out_path))
    plt.close(fig)


def finalise_figure_outputs() -> pd.DataFrame:
    """Write the v5 figure manifest, paper PNGs, and contact sheet."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ALL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if PAPER_FIG_DIR.exists():
        shutil.rmtree(PAPER_FIG_DIR)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for entry in PUB_FIGURE_MANIFEST:
        name = str(entry["name"])
        slot, caption_stub = SLOT_MAP.get(name, ("", ""))
        png_path = Path(entry["png_path"])
        paper_path = PAPER_FIG_DIR / _slot_filename(slot) if slot else Path("")
        rows.append(
            {
                "slot": slot,
                "name": name,
                "group": str(entry["group"]),
                "columns": str(entry["columns"]),
                "png_relpath": str(png_path.relative_to(OUT_DIR)),
                "paper_png_relpath": str(paper_path.relative_to(OUT_DIR)) if slot else "",
                "caption_stub": caption_stub,
                "title_original": str(entry.get("title_original", "")),
            }
        )
    manifest_df = pd.DataFrame(
        rows,
        columns=[
            "slot",
            "name",
            "group",
            "columns",
            "png_relpath",
            "paper_png_relpath",
            "caption_stub",
            "title_original",
        ],
    )
    manifest_df.to_csv(FIG_DIR / "figure_manifest.csv", index=False)

    used_slots = set()
    for _, row in manifest_df.iterrows():
        slot = str(row["slot"])
        if not slot:
            continue
        if slot in used_slots:
            raise RuntimeError(f"Duplicate figure slot in manifest: {slot}")
        used_slots.add(slot)
        src = OUT_DIR / str(row["png_relpath"])
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, PAPER_FIG_DIR / _slot_filename(slot))

    expected_slots = {slot for slot, _caption in SLOT_MAP.values() if slot}
    if used_slots != expected_slots:
        missing = sorted(expected_slots - used_slots)
        extra = sorted(used_slots - expected_slots)
        raise RuntimeError(f"Submission slot mismatch; missing={missing}, extra={extra}")
    _build_contact_sheet(manifest_df, FIG_DIR / "contact_sheet.png")
    return manifest_df


def reset_figure_outputs() -> None:
    """Remove existing figure files before a v5 run."""
    PUB_FIGURE_MANIFEST.clear()
    for target_dir in (ALL_FIG_DIR, PAPER_FIG_DIR, FIG_DIR / "main", FIG_DIR / "supplementary", FIG_DIR / "submission"):
        if target_dir.exists():
            shutil.rmtree(target_dir)
    ALL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in (FIG_DIR / "figure_manifest.csv", FIG_DIR / "contact_sheet.png"):
        if stale.exists():
            stale.unlink()


def reset_table_outputs() -> None:
    """Remove existing generated table files before writing current results."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    for table_file in TAB_DIR.iterdir():
        if table_file.is_file() and table_file.suffix.lower() in {".csv", ".json", ".tex"}:
            table_file.unlink()


def plot_main_b10_vs_delta_t(
    delta_grid: np.ndarray,
    picm_l_pred: Dict[str, object],
    picm_c_pred: Dict[str, object],
    lognorm_pred: Dict[str, object],
    out_dir: Path,
    manifest: List[Dict[str, str]],
    mission_cycle_refs: Optional[Sequence[float]] = None,
) -> None:
    """Main figure: regime-aware B10 comparison across Delta T."""
    d = _validate_delta_t_values(delta_grid)
    mission_refs = [float(v) for v in (MISSION_CYCLE_TARGETS if mission_cycle_refs is None else mission_cycle_refs)]
    b10_l = np.asarray(picm_l_pred["b10_samples"], dtype=float)
    b10_c = np.asarray(picm_c_pred["b10_samples"], dtype=float)
    b10_n = np.asarray(lognorm_pred["b10_samples"], dtype=float)

    def q(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q025, q50, q975 = np.quantile(x, [0.025, 0.5, 0.975], axis=0)
        return q025, q50, q975

    l_lo, l_md, l_hi = q(b10_l)
    c_lo, c_md, c_hi = q(b10_c)
    n_lo, n_md, n_hi = q(b10_n)

    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL, 4.05))
    _add_regime_bands(ax, float(np.min(d)), float(np.max(d)))

    ax.fill_between(d, l_lo, l_hi, color=PLOT_COLORS["bayes_band"], alpha=0.28, label="PICM-L 95% CrI")
    ax.plot(d, l_md, color=PLOT_COLORS["bayes_fit"], lw=1.6, label="PICM-L median")
    ax.plot(d, c_md, color=PLOT_COLORS["picm_c"], lw=1.4, ls="-.", label="PICM-C median")
    ax.plot(d, n_md, color=PLOT_COLORS["posterior_density"], lw=1.4, ls="--", label="Lognormal AFT median")

    for i, cyc in enumerate(mission_refs):
        ax.axhline(cyc, color="#7c8796", lw=0.85, ls=":", alpha=0.75,
                   label=f"Mission reference ({int(cyc)} cycles)" if i == 0 else "_nolegend_")

    _draw_reference_markers(ax, REFERENCE_MARKERS, annotate=False)
    ax.axvline(LOWEST_FAILURE_OBSERVED_STRESS, color=PLOT_COLORS["aux"], lw=0.9, ls="--", alpha=0.75)

    ax.set_xlabel("Thermal excursion Delta T (deg C)")
    ax.set_ylabel("Predicted B10 life (cycles)")
    ax.set_xlim(float(np.min(d)), float(np.max(d)))
    ax.set_yscale("log")
    style_axis(ax)
    _add_regime_labels(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=3,
        fontsize=6.9,
        rect_top=0.93,
        legend_y=0.982,
        columnspacing=0.85,
        handlelength=1.7,
    )
    save_figure_dual(
        fig,
        out_dir,
        "figure_b10_vs_deltaT_regime",
        manifest,
        "Main figure",
        (
            "B10 versus Delta T with PICM-L uncertainty, sensitivity-model medians, "
            "mission references, and regime-aware shading for extrapolative, "
            "censoring-dominated, and failure-observed domains."
        ),
    )
    plt.close(fig)


def plot_main_b10_density_10c_15c(
    pred_picm_l_targets: Dict[str, object],
    pred_picm_c_targets: Dict[str, object],
    pred_lognorm_targets: Dict[str, object],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary figure: smooth B10 density panels at 10 C and 15 C."""
    dvals = np.asarray(pred_picm_l_targets["delta_t_values"], dtype=float)
    idx_10 = int(np.argmin(np.abs(dvals - 10.0)))
    idx_15 = int(np.argmin(np.abs(dvals - 15.0)))
    panels = [(idx_10, 10.0), (idx_15, 15.0)]

    model_samples = [
        ("PICM-L", np.asarray(pred_picm_l_targets["b10_samples"], dtype=float), PLOT_COLORS["bayes_fit"]),
        ("PICM-C", np.asarray(pred_picm_c_targets["b10_samples"], dtype=float), PLOT_COLORS["picm_c"]),
        ("Lognormal AFT", np.asarray(pred_lognorm_targets["b10_samples"], dtype=float), PLOT_COLORS["posterior_density"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_TWO_COL, 3.45), sharey=True)
    for ax, (idx, d_target) in zip(axes, panels):
        all_vals = np.hstack([np.clip(s[:, idx], 1e-12, np.inf) for _, s, _ in model_samples])
        lo = max(float(np.quantile(all_vals, 0.003)), 1e-6)
        hi = float(np.quantile(all_vals, 0.997))
        xg = np.logspace(np.log10(lo), np.log10(hi), 400)

        for label, arr, color in model_samples:
            vals = np.clip(arr[:, idx], 1e-12, np.inf)
            q025, q50, q975 = np.quantile(vals, [0.025, 0.5, 0.975])
            kde = stats.gaussian_kde(np.log10(vals))
            density = kde(np.log10(xg)) / (xg * np.log(10.0))
            ax.plot(xg, density, color=color, lw=1.5, label=label)
            ax.axvline(q50, color=color, lw=1.1, alpha=0.9)

        ax.set_xscale("log")
        ax.set_xlabel("B10 life (cycles)")
        panel_tag = "a" if int(d_target) == 10 else "b"
        ax.set_title(f"({panel_tag}) Delta T = {int(d_target)} deg C", fontsize=8.5, fontweight="regular")
        style_axis(ax)
    axes[0].set_ylabel("Density")
    h, l = dedup_legend(*axes[0].get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=3,
        fontsize=6.9,
        rect_top=0.95,
        legend_y=0.985,
        columnspacing=0.85,
        handlelength=1.7,
    )
    save_figure_dual(
        fig,
        out_dir,
        "figure_b10_density_10C_15C",
        manifest,
        "Supplementary",
        (
            "B10 density comparison at 10 C and 15 C for PICM-L, PICM-C, "
            "and lognormal AFT with model-specific median and 95% interval markers."
        ),
    )
    plt.close(fig)


def plot_high_deltaT_extrapolation_ci(
    pred_picm_l_high_grid: Dict[str, object],
    high_dt_posterior: pd.DataFrame,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary figure: high-side PICM-L extrapolation with posterior interval."""
    d = np.asarray(pred_picm_l_high_grid["delta_t_values"], dtype=float)
    b10 = np.asarray(pred_picm_l_high_grid["b10_samples"], dtype=float)
    lo, md, hi = np.quantile(b10, [0.025, 0.5, 0.975], axis=0)

    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL, 3.35))
    ax.axvspan(float(np.min(d)), OBSERVED_DOMAIN_MAX, color="#dcfce7", alpha=0.10, zorder=0)
    ax.axvspan(OBSERVED_DOMAIN_MAX, float(np.max(d)), color="#fee2e2", alpha=0.18, zorder=0)
    ax.axvline(OBSERVED_DOMAIN_MAX, color=PLOT_COLORS["aux"], lw=0.9, ls="--", alpha=0.8)

    ax.fill_between(d, lo, hi, color=PLOT_COLORS["bayes_band"], alpha=0.46, label="PICM-L 95% CrI")
    ax.plot(d, md, color=PLOT_COLORS["bayes_fit"], lw=1.65, label="PICM-L posterior median")

    targets = high_dt_posterior.sort_values("delta_T_C").reset_index(drop=True)
    tx = targets["delta_T_C"].to_numpy(dtype=float)
    ty = targets["B10_median_cycles"].to_numpy(dtype=float)
    tlo = targets["B10_q025_cycles"].to_numpy(dtype=float)
    thi = targets["B10_q975_cycles"].to_numpy(dtype=float)
    yerr = np.vstack([ty - tlo, thi - ty])
    ax.errorbar(
        tx,
        ty,
        yerr=yerr,
        fmt="o",
        ms=4.2,
        lw=1.1,
        capsize=3.0,
        color=PLOT_COLORS["failed"],
        ecolor=PLOT_COLORS["failed"],
        label="175/200 deg C posterior summaries",
        zorder=5,
    )
    for x, y in zip(tx, ty):
        ax.annotate(
            f"{int(round(x))} deg C",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.0,
            color=PLOT_COLORS["aux"],
        )

    y_top = float(np.nanmax(hi))
    ax.text(122.0, y_top * 0.78, "tested domain", fontsize=7.0, color=PLOT_COLORS["aux"])
    ax.text(161.0, y_top * 0.78, "above tested range", fontsize=7.0, color=PLOT_COLORS["aux"])

    ax.set_xlabel("Protocol-defined Delta T (deg C)")
    ax.set_ylabel("Predicted B10 life (cycles)")
    ax.set_xlim(float(np.min(d)), float(np.max(d)))
    ax.set_yscale("log")
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=3,
        fontsize=6.9,
        rect_top=0.91,
        legend_y=0.985,
        columnspacing=0.85,
        handlelength=1.7,
    )
    save_figure_dual(
        fig,
        out_dir,
        "figure_high_deltaT_extrapolation_CI",
        manifest,
        "Supplementary",
        (
            "High-side PICM-L extrapolation from 100 C to 200 C with posterior "
            "median, 95% credible interval, and 175/200 C target summaries; "
            "values above 150 C are model-conditional."
        ),
    )
    plt.close(fig)


def plot_prior_sensitivity_b10_35c(
    b10_35_by_prior: Dict[str, np.ndarray],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: B10(35C) overlay under alternative prior settings for PICM-L."""
    if not b10_35_by_prior:
        warnings.warn("No prior-sensitivity draws provided for B10(35C); skipping figure.")
        return

    color_map = {"baseline": "#1d4ed8", "diffuse": "#b91c1c", "conservative": "#047857"}
    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.8))
    all_vals = np.hstack([np.asarray(v, dtype=float) for v in b10_35_by_prior.values()])
    x_lo = max(np.quantile(all_vals, 0.005), 1e-9)
    x_hi = np.quantile(all_vals, 0.995)
    xg = np.logspace(np.log10(x_lo), np.log10(x_hi), 350)

    for key in PRIOR_SENSITIVITY_KEYS:
        if key not in b10_35_by_prior:
            continue
        vals = np.asarray(b10_35_by_prior[key], dtype=float)
        kde = stats.gaussian_kde(vals)
        y = kde(xg)
        c = color_map.get(key, "#4b5563")
        ax.plot(xg, y, lw=2.0, color=c, label=f"{key} prior")
        med = float(np.quantile(vals, 0.5))
        ax.axvline(med, color=c, lw=1.2, ls="--")

    ax.set_xscale("log")
    ax.set_xlabel("B10 life at Delta T = 35 deg C (cycles)")
    ax.set_ylabel("Density")
    style_axis(ax)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.985], pad=0.6)
    save_figure_dual(
        fig,
        out_dir,
        "supp_prior_sensitivity_b10_35C",
        manifest,
        "Supplementary",
        "PICM-L reference-case B10 density overlay at Delta T = 35 deg C for baseline, diffuse, and conservative prior settings.",
    )
    plt.close(fig)

def plot_main_failure_fraction_calibration(
    failure_fraction_check: pd.DataFrame,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main figure: observed-domain failure-fraction calibration for PICM-L."""
    ff = failure_fraction_check.copy().sort_values("delta_T").reset_index(drop=True)
    stress = ff["delta_T"].to_numpy(dtype=float)
    obs = ff["observed_failure_fraction"].to_numpy(dtype=float)
    pred = ff["predicted_median_failure_fraction"].to_numpy(dtype=float)
    lo = ff["predicted_pi_2.5%"].to_numpy(dtype=float)
    hi = ff["predicted_pi_97.5%"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.75))
    ax.fill_between(stress, lo, hi, color=PLOT_COLORS["bayes_band"], alpha=0.55, label="PICM-L posterior-predictive 95% PI")
    ax.plot(stress, pred, marker="s", color=PLOT_COLORS["bayes_fit"], lw=2.0, label="Predicted median fraction")
    ax.plot(stress, obs, marker="o", color=PLOT_COLORS["failed"], lw=1.7, label="Observed fraction")

    if {"observed_failed", "observed_total"}.issubset(set(ff.columns)):
        for _, r in ff.iterrows():
            s = float(r["delta_T"])
            txt = f"{int(r['observed_failed'])}/{int(r['observed_total'])}"
            # Only annotate black counts where they carry unique information.
            if s in (25.0, 50.0, 75.0, 150.0):
                y_txt = max(float(r["observed_failure_fraction"]) + 0.045, 0.06)
                x_txt = s
                ha = "center"
                if s == 150.0:
                    x_txt = s - 2.0
                    ha = "right"
                ax.text(x_txt, min(y_txt, 0.98), txt, ha=ha, va="bottom", fontsize=8.1, color=PLOT_COLORS["aux"])

    for s, p in zip(stress, pred):
        # Only annotate predicted medians when they differ visibly from the observed fractions.
        if s in (75.0, 150.0):
            x_txt = float(s)
            y_txt = min(float(p) + 0.055, 0.98)
            ha = "center"
            if s == 150.0:
                x_txt = float(s) - 2.0
                y_txt = min(float(p) + 0.04, 0.90)
                ha = "right"
            ax.text(x_txt, y_txt, f"{p:.2f}", ha=ha, va="bottom", fontsize=7.8, color=PLOT_COLORS["bayes_fit"])

    ax.set_xlabel("Delta T (deg C)")
    ax.set_ylabel("Failure fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(stress)
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=1,
        fontsize=7.3,
        rect_top=0.78,
        legend_y=0.992,
        columnspacing=0.9,
        handlelength=1.8,
    )
    save_figure_dual(
        fig,
        out_dir,
        "figure_failure_fraction_calibration",
        manifest,
        "Main figure",
        (
            "Observed-domain failure-fraction calibration for PICM-L showing "
            "posterior-predictive 95% intervals, predictive medians, and observed "
            "stress-wise failure fractions."
        ),
    )
    plt.close(fig)


def plot_main_loso_observed_vs_pred(
    loso_table: pd.DataFrame,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary figure: LOSO observed-vs-predicted failure fraction with uncertainty bars."""
    if loso_table.empty:
        warnings.warn("LOSO table is empty; skipping LOSO figure.")
        return
    chk = loso_table.sort_values("held_out_stress").reset_index(drop=True)
    x = chk["observed_failure_fraction"].to_numpy(dtype=float)
    y = chk["predicted_failure_fraction"].to_numpy(dtype=float)
    ylo = chk["predicted_pi_2_5"].to_numpy(dtype=float)
    yhi = chk["predicted_pi_97_5"].to_numpy(dtype=float)
    stress = chk["held_out_stress"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.8))
    ax.errorbar(
        x,
        y,
        yerr=[y - ylo, yhi - y],
        fmt="o",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        capsize=3,
        lw=1.4,
        ms=6,
        label="LOSO prediction (95% PI)",
    )
    ax.plot([0, 1], [0, 1], color=PLOT_COLORS["mle"], ls="--", lw=1.3, label="45-degree reference")
    for xx, yy, ss, ofail, otot in zip(x, y, stress, chk["observed_failed"].to_numpy(int), chk["observed_total"].to_numpy(int)):
        ax.text(float(xx) + 0.013, float(yy) + 0.012, f"{int(ss)} deg C ({ofail}/{otot})", fontsize=8.2, color=PLOT_COLORS["aux"])

    ax.set_xlabel("Observed failure fraction (held-out stress)")
    ax.set_ylabel("Predicted failure fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)
    ax.legend(loc="upper left")
    save_figure_dual(
        fig,
        out_dir,
        "supp_leave_one_stress_out_mainstyle",
        manifest,
        "Supplementary",
        "Supplementary LOSO observed-vs-predicted failure-fraction plot with predictive uncertainty.",
    )
    plt.close(fig)


def plot_main_stress_life_logy(
    data: ReliabilityData,
    pred_picm_l_grid: Dict[str, object],
    pred_picm_c_grid: Dict[str, object],
    pred_lognorm_grid: Dict[str, object],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main figure: stress-life data with regime-aware extrapolation delineation."""
    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL, 4.05))
    rng = np.random.default_rng(SEED_PRED + 66)
    x = data.df["delta_T"].to_numpy(dtype=float)
    xj = x + rng.normal(0.0, 0.85, size=x.size)
    y = data.df["cycles"].to_numpy(dtype=float)
    e = data.df["event"].to_numpy(dtype=int)

    fail_by_stress = data.df.groupby("delta_T")["event"].sum()
    failing_stresses = fail_by_stress[fail_by_stress > 0].index.to_numpy(dtype=float)
    low_fail_stress = float(np.min(failing_stresses)) if failing_stresses.size > 0 else LOWEST_FAILURE_OBSERVED_STRESS

    _add_regime_bands(ax, float(np.min(B10_GRID_DELTA_T)), float(OBSERVED_DOMAIN_MAX))
    ax.axvline(low_fail_stress, color=PLOT_COLORS["aux"], lw=0.9, ls="--", alpha=0.75)

    ax.scatter(xj[e == 1], y[e == 1], s=22, c=PLOT_COLORS["failed"], alpha=0.78, label="Failed units")
    ax.scatter(
        xj[e == 0],
        y[e == 0],
        s=24,
        facecolors="white",
        edgecolors=PLOT_COLORS["censored_edge"],
        linewidths=1.0,
        label="Right-censored units",
    )

    d = np.asarray(pred_picm_l_grid["delta_t_values"], dtype=float)
    l_sum = pred_picm_l_grid["median_summary"].set_index("delta_T")
    c_sum = pred_picm_c_grid["median_summary"].set_index("delta_T")
    n_sum = pred_lognorm_grid["median_summary"].set_index("delta_T")

    l_md = l_sum.loc[d, "medianlife_q50"].to_numpy(dtype=float)
    l_lo = l_sum.loc[d, "medianlife_q025"].to_numpy(dtype=float)
    l_hi = l_sum.loc[d, "medianlife_q975"].to_numpy(dtype=float)
    c_md = c_sum.loc[d, "medianlife_q50"].to_numpy(dtype=float)
    n_md = n_sum.loc[d, "medianlife_q50"].to_numpy(dtype=float)

    ax.fill_between(d, l_lo, l_hi, color=PLOT_COLORS["bayes_band"], alpha=0.28, label="PICM-L 95% CrI")
    ax.plot(d, l_md, color=PLOT_COLORS["bayes_fit"], lw=1.6, label="PICM-L median life")
    ax.plot(d, c_md, color=PLOT_COLORS["picm_c"], lw=1.4, ls="-.", label="PICM-C median life")
    ax.plot(d, n_md, color=PLOT_COLORS["posterior_density"], lw=1.4, ls="--", label="Lognormal AFT median life")

    ax.set_yscale("log")
    ax.set_xlim(float(np.min(B10_GRID_DELTA_T)), float(OBSERVED_DOMAIN_MAX) + 2.0)
    ax.set_xlabel("Thermal excursion Delta T (deg C)")
    ax.set_ylabel("Cycles to failure")
    style_axis(ax)

    _draw_reference_markers(ax, REFERENCE_MARKERS, annotate=False)
    _add_regime_labels(ax)

    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=3,
        fontsize=6.9,
        rect_top=0.93,
        legend_y=0.982,
        columnspacing=0.85,
        handlelength=1.7,
    )
    save_figure_dual(
        fig,
        out_dir,
        "figure_stress_life_regime",
        manifest,
        "Main figure",
        (
            "Stress-life scatter with censoring marks, PICM-L uncertainty, "
            "sensitivity-model median curves, and regime-aware shading for "
            "extrapolative, censoring-dominated, and failure-observed domains."
        ),
    )
    plt.close(fig)


def km_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kaplan-Meier step points and censor times."""
    order = np.argsort(times)
    t = times[order]
    e = events[order]
    uniq = np.unique(t)
    at_risk = len(t)
    surv = 1.0
    fail_t = []
    fail_s = []
    for u in uniq:
        d = int(np.sum((t == u) & (e == 1)))
        c = int(np.sum((t == u) & (e == 0)))
        if d > 0 and at_risk > 0:
            surv *= (1.0 - d / at_risk)
            fail_t.append(float(u))
            fail_s.append(float(surv))
        at_risk -= (d + c)
    censor_t = np.sort(t[e == 0])
    return np.array(fail_t), np.array(fail_s), censor_t


def km_surv_at(censor_t: np.ndarray, fail_t: np.ndarray, fail_s: np.ndarray) -> np.ndarray:
    """KM survival value at censor times (post-step convention)."""
    out = np.ones_like(censor_t, dtype=float)
    for i, ct in enumerate(censor_t):
        idx = np.searchsorted(fail_t, ct, side="right") - 1
        out[i] = 1.0 if idx < 0 else fail_s[idx]
    return out


def make_plots(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    bayes_samples: np.ndarray,
    chains: np.ndarray,
    profile_res: Dict[str, object],
    bartlett_factor: float,
    post_pred: Dict[str, pd.DataFrame],
    delta_t_use: float,
    out_dir: Path,
    b10_mle_cycles: Optional[float] = None,
    b10_firth_cycles: Optional[float] = None,
) -> None:
    """Generate all required main and supplementary figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_PRED)

    c_fail = "#1f1f1f"
    c_cens = "#4d4d4d"
    c_model = "#1f4e79"
    c_mle = "#e07a1f"
    c_firth = "#5f0f99"
    c_band = "#b7d4e8"
    c_km = "#8c2d2d"
    c_post = "#0b6e4f"
    c_prior = "#8c2d2d"

    k_s = bayes_samples[:, 0]
    b1_s = bayes_samples[:, 1]
    b0_s = bayes_samples[:, 2]
    k_mle, b1_mle, b0_mle = mle_theta

    # Main Figure 1: Stress-life scatter + posterior median curve/band
    fig, ax = plt.subplots(figsize=(8, 5))
    x_obs = np.log(data.df["delta_T"].to_numpy())
    y_obs = data.df["cycles"].to_numpy()
    ev = data.df["event"].to_numpy()
    ax.scatter(x_obs[ev == 1], y_obs[ev == 1], s=40, c=c_fail, alpha=0.9, label="Failed units")
    ax.scatter(
        x_obs[ev == 0],
        y_obs[ev == 0],
        s=40,
        facecolors="none",
        edgecolors=c_cens,
        linewidths=1.2,
        label="Right-censored units",
    )
    xg = np.linspace(np.min(x_obs) - 0.08, np.max(x_obs) + 0.08, 140)
    eta_grid = np.exp(b0_s[:, None] - b1_s[:, None] * xg[None, :])
    med_grid = eta_grid * (np.log(2.0) ** (1.0 / k_s[:, None]))
    q_lo, q_md, q_hi = np.quantile(med_grid, [0.025, 0.5, 0.975], axis=0)
    ax.fill_between(xg, q_lo, q_hi, color=c_band, alpha=0.55, label="95% credible band")
    ax.plot(xg, q_md, color=c_model, lw=2.2, label="PICM-L posterior median")
    eta_mle_grid = np.exp(b0_mle - b1_mle * xg)
    med_mle_grid = eta_mle_grid * (np.log(2.0) ** (1.0 / k_mle))
    ax.plot(xg, med_mle_grid, color=c_mle, lw=2.0, ls="--", label="PICM-L MLE median")
    ax.set_xlabel("log(Delta T [C])")
    ax.set_ylabel("Cycles")
    ax.set_title("Stress-Life Data with PICM-L Fits")
    ax.set_ylim(0.0, 1.08 * np.max(y_obs))
    style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    h, l = dedup_legend(h, l)
    ax.legend(h, l, loc="upper right", ncol=1)
    fig.tight_layout()
    save_pub_figure(fig, "main_stress_life", _figure_group_from_dir(out_dir), columns="double")
    plt.close(fig)

    # Main Figure 2: Posterior predictive survival curves by stress with KM + censor marks
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8), sharey=True)
    axes = axes.ravel()
    subs_idx = rng.choice(bayes_samples.shape[0], size=min(5000, bayes_samples.shape[0]), replace=False)
    subs = bayes_samples[subs_idx, :]
    for i, s in enumerate(data.stress_levels):
        ax = axes[i]
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        e = data.event[idx]
        x = np.log(s)

        x_max = max(np.max(y) * 1.2, 1.0)
        yg = np.linspace(1e-3, x_max, 220)
        k_sub = subs[:, 0][:, None]
        eta_sub = np.exp(subs[:, 2][:, None] - subs[:, 1][:, None] * x)
        surv = np.exp(-((yg[None, :] / eta_sub) ** k_sub))
        s_lo, s_md, s_hi = np.quantile(surv, [0.025, 0.5, 0.975], axis=0)
        ax.fill_between(yg, s_lo, s_hi, color=c_band, alpha=0.55, label="95% credible band" if i == 0 else "_nolegend_")
        ax.plot(yg, s_md, color=c_model, lw=2.1, label="Posterior median" if i == 0 else "_nolegend_")
        eta_mle = np.exp(b0_mle - b1_mle * x)
        s_mle = np.exp(-((yg / eta_mle) ** k_mle))
        ax.plot(yg, s_mle, color=c_mle, lw=1.8, ls="--", label="PICM-L MLE" if i == 0 else "_nolegend_")

        ft, fs, ct = km_curve(y, e)
        if ft.size > 0:
            ax.step(
                np.r_[0.0, ft],
                np.r_[1.0, fs],
                where="post",
                color=c_km,
                lw=1.7,
                label="Kaplan-Meier" if i == 0 else "_nolegend_",
            )
        if ct.size > 0:
            cs = km_surv_at(ct, ft, fs)
            ax.plot(
                ct,
                cs,
                linestyle="None",
                marker="+",
                color=c_km,
                ms=7,
                mew=1.3,
                label="Censor marks" if i == 0 else "_nolegend_",
            )
        if int(np.sum(e)) == 0:
            ax.text(0.04, 0.08, "All units censored", transform=ax.transAxes, fontsize=8.5, color=c_cens)

        ax.set_title(f"Delta T = {int(s)} C ({int(np.sum(e))}/8 failed)")
        ax.set_xlabel("Cycles")
        if i % 3 == 0:
            ax.set_ylabel("Survival")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, x_max)
        style_axis(ax)
    handles, labels = [], []
    for ax_i in axes:
        h, l = ax_i.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    handles, labels = dedup_legend(handles, labels)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Survival by Stress: Posterior, MLE and Kaplan-Meier", y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.955])
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_pub_figure(fig, "main_survival_panels", _figure_group_from_dir(out_dir), columns="double")
    plt.close(fig)

    # Main Figure 3: Use-condition B10 posterior distribution
    b10_use = post_pred["use_b10_samples"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(b10_use, bins=40, density=True, color=c_band, alpha=0.85, edgecolor="white", label="Posterior samples")
    if len(b10_use) > 50:
        kde = stats.gaussian_kde(b10_use)
        xk = np.linspace(np.min(b10_use), np.max(b10_use), 300)
        ax.plot(xk, kde(xk), color=c_model, lw=2.2, label="KDE")
    q025, q50, q975 = np.quantile(b10_use, [0.025, 0.5, 0.975])
    ax.axvspan(q025, q975, color="#d9d9d9", alpha=0.35, label="95% credible interval")
    ax.axvline(q50, color=c_fail, lw=2.0, label="Posterior median")
    ax.axvline(q025, color=c_fail, ls="--", lw=1.2)
    ax.axvline(q975, color=c_fail, ls="--", lw=1.2)
    if b10_mle_cycles is not None:
        ax.axvline(b10_mle_cycles, color=c_mle, lw=1.8, ls="-.", label="MLE B10")
    if b10_firth_cycles is not None:
        ax.axvline(b10_firth_cycles, color=c_firth, lw=1.8, ls=":", label="Firth B10")
    ax.set_xlabel(f"Use-condition B10 cycles (Delta T = {delta_t_use:g} C)")
    ax.set_ylabel("Density")
    ax.set_title("Use-Condition B10 Distribution and Point Estimates")
    style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    h, l = dedup_legend(h, l)
    ax.legend(h, l, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, "main_use_b10_posterior", _figure_group_from_dir(out_dir), columns="single")
    plt.close(fig)

    # Supplementary: Trace plots
    param_names = ["k", "beta1", "beta0"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    chain_colors = ["#1f4e79", "#2a9d8f", "#8c2d2d"]
    for j in range(3):
        ax = axes[j]
        for c in range(chains.shape[0]):
            ax.plot(
                chains[c, :, j],
                lw=0.75,
                alpha=0.85,
                color=chain_colors[c % len(chain_colors)],
                label=f"Chain {c+1}" if j == 0 else None,
            )
        ax.set_ylabel(param_names[j])
        style_axis(ax)
    axes[-1].set_xlabel("Post-burn iteration")
    if chains.shape[0] > 0:
        axes[0].legend(loc="upper right", ncol=3, frameon=True)
    fig.suptitle("PICM-L Posterior Trace Plots", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_pub_figure(fig, "supp_trace_plots", _figure_group_from_dir(out_dir), columns="double")
    plt.close(fig)

    # Supplementary: posterior marginals + prior overlays
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    names = ["k", "beta1", "beta0"]
    priors = [
        lambda x: stats.gamma.pdf(x, a=9.0, scale=1.0 / 3.0),
        lambda x: stats.truncnorm.pdf(x, (0.5 - 2.5) / 0.8, np.inf, loc=2.5, scale=0.8),
        lambda x: stats.norm.pdf(x, loc=18.0, scale=4.0),
    ]
    for j in range(3):
        ax = axes[j]
        x = bayes_samples[:, j]
        ax.hist(x, bins=45, density=True, alpha=0.7, color="#b9d7cf", edgecolor="white", label="Posterior")
        kde = stats.gaussian_kde(x)
        xx = np.linspace(np.min(x), np.max(x), 400)
        ax.plot(xx, kde(xx), color=c_post, lw=2.2, label="Posterior KDE")
        ax.set_title(f"Posterior: {names[j]}")
        ax.set_xlabel(names[j])
        ax.set_ylabel("Density")
        style_axis(ax)
        h, l = ax.get_legend_handles_labels()
        h, l = dedup_legend(h, l)
        ax.legend(h, l, loc="upper right")

        ax2 = axes[j + 3]
        xx2 = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 400)
        ax2.plot(xx2, priors[j](xx2), color=c_prior, lw=1.8, label="Prior")
        ax2.plot(xx, kde(xx), color=c_post, lw=2.0, label="Posterior KDE")
        ax2.set_title(f"Prior vs Posterior: {names[j]}")
        ax2.set_xlabel(names[j])
        ax2.set_ylabel("Density")
        style_axis(ax2)
        h2, l2 = ax2.get_legend_handles_labels()
        h2, l2 = dedup_legend(h2, l2)
        ax2.legend(h2, l2, loc="upper right")
    fig.tight_layout()
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_pub_figure(fig, "supp_marginals_and_prior_posterior", _figure_group_from_dir(out_dir), columns="double")
    plt.close(fig)

    # Supplementary: Weibull probability plots by stress
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, s in enumerate(data.stress_levels):
        ax = axes[i]
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        e = data.event[idx]
        ft, fs, _ct = km_curve(y, e)
        if ft.size > 1:
            mask = (fs > 0.0) & (fs < 1.0)
            xx = np.log(ft[mask])
            yy = np.log(-np.log(fs[mask]))
            ax.scatter(xx, yy, c=c_fail, s=28, alpha=0.9, label="KM failure points")
        eta_mle = np.exp(b0_mle - b1_mle * np.log(s))
        xline = np.linspace(np.log(max(np.min(y) * 0.9, 1e-3)), np.log(np.max(y) * 1.1), 150)
        yline = k_mle * (xline - np.log(eta_mle))
        ax.plot(xline, yline, color=c_model, lw=2.0, label="PICM-L MLE line")
        ax.set_title(f"Delta T = {int(s)} C")
        if i % 3 == 0:
            ax.set_ylabel("log(-log(S))")
        if i >= 3:
            ax.set_xlabel("log(Cycles)")
        style_axis(ax)
    handles, labels = [], []
    for ax_i in axes:
        h, l = ax_i.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    handles, labels = dedup_legend(handles, labels)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Weibull Probability Plots by Stress", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_pub_figure(fig, "supp_weibull_probability_plots", _figure_group_from_dir(out_dir), columns="double")
    plt.close(fig)

    # Supplementary: profile-likelihood curve (uncorrected + Bartlett-corrected)
    lg = profile_res["logpsi_grid"]
    lr = profile_res["lr_grid"]
    q95 = profile_res["q95"]
    fig, ax = plt.subplots(figsize=(8, 5))
    b10_grid = np.exp(lg)
    lr_corr = lr / bartlett_factor
    ax.plot(b10_grid, lr, color=c_model, lw=2.2, label="Uncorrected LR profile")
    ax.plot(b10_grid, lr_corr, color=c_km, lw=2.0, ls="--", label="Bartlett-corrected LR")
    ax.axhline(q95, color=c_fail, ls=":", lw=1.6, label="Chi-square threshold (95%, df=1)")
    ax.set_xlabel("Reference-case B10 (cycles)")
    ax.set_ylabel("Likelihood-ratio statistic")
    ax.set_title("Profile Likelihood for Use-Condition B10")
    style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    h, l = dedup_legend(h, l)
    ax.legend(h, l, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, "supp_profile_likelihood_b10", _figure_group_from_dir(out_dir), columns="single")
    plt.close(fig)

    # Supplementary: observed vs posterior-predictive failure fraction by stress
    fig, ax = plt.subplots(figsize=(8, 5))
    stress = data.stress_levels
    obs_frac = []
    pred_med = []
    pred_lo = []
    pred_hi = []
    for s in stress:
        idx = data.stress_to_idx[float(s)]
        dsub = data.df.iloc[idx]
        obs_frac.append(float(np.mean(dsub["event"].to_numpy(dtype=int))))

        # stress-specific runout: max observed censor if present else max observed cycle
        cens = dsub[dsub["event"] == 0]["cycles"].to_numpy(dtype=float)
        runout = float(np.max(cens)) if cens.size > 0 else float(np.max(dsub["cycles"].to_numpy(dtype=float)))

        x = np.log(s)
        eta = np.exp(b0_s - b1_s * x)
        p_fail = 1.0 - np.exp(-((runout / eta) ** k_s))
        # finite-sample posterior-predictive fraction for n=8
        pred_counts = rng.binomial(8, np.clip(p_fail, 1e-12, 1 - 1e-12))
        frac = pred_counts / 8.0
        ql, qm, qu = np.quantile(frac, [0.025, 0.5, 0.975])
        pred_lo.append(float(ql))
        pred_med.append(float(qm))
        pred_hi.append(float(qu))

    stress = np.asarray(stress, dtype=float)
    obs_frac = np.asarray(obs_frac, dtype=float)
    pred_med = np.asarray(pred_med, dtype=float)
    pred_lo = np.asarray(pred_lo, dtype=float)
    pred_hi = np.asarray(pred_hi, dtype=float)

    ax.fill_between(stress, pred_lo, pred_hi, color=c_band, alpha=0.45, label="Posterior-predictive 95% PI")
    ax.plot(stress, pred_med, marker="s", color=c_model, lw=1.8, label="Predicted median failure fraction")
    ax.plot(stress, obs_frac, marker="o", color=c_fail, lw=1.4, label="Observed failure fraction")
    ax.set_xlabel("Delta T (deg C)")
    ax.set_ylabel("Failure fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(stress)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_title("Observed vs Posterior-Predictive Failure Fractions")
    style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    h, l = dedup_legend(h, l)
    ax.legend(h, l, loc="lower right")
    fig.tight_layout()
    save_pub_figure(fig, "supp_failure_fraction_check", _figure_group_from_dir(out_dir), columns="single")
    plt.close(fig)


# Publication-grade figure set (main + supplementary)
def plot_main_stress_life_comparison(
    data: ReliabilityData,
    bayes_samples: np.ndarray,
    mle_theta: np.ndarray,
    firth_theta: np.ndarray,
    picm_c_theta: np.ndarray,
    delta_t_use: float,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main Figure 1: Stress-life comparison with Bayesian and frequentist fits."""
    c_fail = PLOT_COLORS["failed"]
    c_cens = PLOT_COLORS["censored_edge"]
    c_bayes = PLOT_COLORS["bayes_fit"]
    c_bayes_band = PLOT_COLORS["bayes_band"]
    c_mle = PLOT_COLORS["mle"]
    c_firth = PLOT_COLORS["firth"]
    c_picmc = PLOT_COLORS["picm_c"]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    rng = np.random.default_rng(SEED_PRED + 11)
    x_obs = np.log(data.df["delta_T"].to_numpy(dtype=float))
    x_obs_j = x_obs + rng.normal(0.0, 0.008, size=x_obs.size)
    y_obs = data.df["cycles"].to_numpy(dtype=float)
    ev = data.df["event"].to_numpy(dtype=int)

    ax.scatter(x_obs_j[ev == 1], y_obs[ev == 1], s=40, c=c_fail, alpha=0.9, label="Failed")
    ax.scatter(
        x_obs_j[ev == 0],
        y_obs[ev == 0],
        s=42,
        facecolors="white",
        edgecolors=c_cens,
        linewidths=1.2,
        label="Right-censored",
        zorder=3,
    )

    xg = np.linspace(np.min(x_obs) - 0.08, np.max(x_obs) + 0.08, 220)
    k_s, b1_s, b0_s = bayes_samples[:, 0], bayes_samples[:, 1], bayes_samples[:, 2]
    eta_g = np.exp(b0_s[:, None] - b1_s[:, None] * xg[None, :])
    med_g = eta_g * (np.log(2.0) ** (1.0 / k_s[:, None]))
    ql, qm, qu = np.quantile(med_g, [0.025, 0.5, 0.975], axis=0)
    ax.fill_between(xg, ql, qu, color=c_bayes_band, alpha=0.55, label="PICM-L 95% CrI")
    ax.plot(xg, qm, color=c_bayes, lw=2.3, label="PICM-L posterior median")

    k_mle, b1_mle, b0_mle = [float(v) for v in mle_theta]
    eta_mle = np.exp(b0_mle - b1_mle * xg)
    med_mle = eta_mle * (np.log(2.0) ** (1.0 / k_mle))
    ax.plot(xg, med_mle, color=c_mle, lw=2.0, ls="--", label="PICM-L MLE")

    k_f, b1_f, b0_f = [float(v) for v in firth_theta]
    eta_f = np.exp(b0_f - b1_f * xg)
    med_f = eta_f * (np.log(2.0) ** (1.0 / k_f))
    ax.plot(xg, med_f, color=c_firth, lw=1.9, ls=(0, (6, 2, 1.8, 2)), label="PICM-L Firth")

    k_c, b1_c, b0_c, b2_c = [float(v) for v in picm_c_theta]
    log_eta_c = b0_c - b1_c * xg + b2_c * (xg - data.xbar) ** 2
    med_c = np.exp(log_eta_c) * (np.log(2.0) ** (1.0 / k_c))
    ax.plot(xg, med_c, color=c_picmc, lw=1.8, ls="-.", label="PICM-C sensitivity")

    x_use = np.log(delta_t_use)
    ax.axvline(x_use, color=PLOT_COLORS["aux"], lw=1.4, ls=":", label=f"Reference case (Delta T = {delta_t_use:g} deg C)")
    ax.text(x_use + 0.01, 0.92 * np.max(y_obs), "Use", fontsize=9, color=PLOT_COLORS["aux"])

    ax.set_xlabel("log(Delta T / deg C)")
    ax.set_ylabel("Cycles to failure")
    ax.set_title("Stress-Life Comparison: Data, Primary Model, and Sensitivity Fits")
    ax.set_ylim(0.0, 1.10 * np.max(y_obs))
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    ax.legend(h, l, loc="upper right", ncol=2, fontsize=8.8)
    save_figure_dual(
        fig,
        out_dir,
        "main_stress_life_comparison",
        manifest,
        "Main figure",
        "Stress-life relation comparing data with Bayesian PICM-L, MLE, Firth, and PICM-C sensitivity.",
    )
    plt.close(fig)


def plot_main_survival_panels(
    data: ReliabilityData,
    bayes_samples: np.ndarray,
    mle_theta: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main Figure 2: stress-wise posterior predictive survival with KM overlays."""
    c_bayes = PLOT_COLORS["bayes_fit"]
    c_band = PLOT_COLORS["bayes_band"]
    c_km = PLOT_COLORS["km_empirical"]
    c_mle = PLOT_COLORS["mle"]
    k_mle, b1_mle, b0_mle = [float(v) for v in mle_theta]

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_TWO_COL, 4.65), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(SEED_PRED + 12)
    subs = bayes_samples[rng.choice(bayes_samples.shape[0], size=min(5000, bayes_samples.shape[0]), replace=False), :]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, s in enumerate(data.stress_levels):
        ax = axes[i]
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        e = data.event[idx]
        x = np.log(s)
        x_max = max(1.15 * np.max(y), 1.0)
        yg = np.linspace(1e-3, x_max, 250)

        k_sub = subs[:, 0][:, None]
        eta_sub = np.exp(subs[:, 2][:, None] - subs[:, 1][:, None] * x)
        surv = np.exp(-((yg[None, :] / eta_sub) ** k_sub))
        ql, qm, qu = np.quantile(surv, [0.025, 0.5, 0.975], axis=0)
        ax.fill_between(yg, ql, qu, color=c_band, alpha=0.58, label="PICM-L 95% CrI" if i == 0 else "_nolegend_")
        ax.plot(yg, qm, color=c_bayes, lw=2.1, label="PICM-L posterior median" if i == 0 else "_nolegend_")

        eta_mle = np.exp(b0_mle - b1_mle * x)
        s_mle = np.exp(-((yg / eta_mle) ** k_mle))
        ax.plot(yg, s_mle, color=c_mle, lw=1.6, ls="--", label="PICM-L MLE" if i == 0 else "_nolegend_")

        ft, fs, ct = km_curve(y, e)
        if ft.size > 0:
            ax.step(np.r_[0.0, ft], np.r_[1.0, fs], where="post", color=c_km, lw=1.7, label="Kaplan-Meier" if i == 0 else "_nolegend_")
        if ct.size > 0:
            cs = km_surv_at(ct, ft, fs)
            ax.plot(ct, cs, linestyle="None", marker="+", color=c_km, ms=6.5, mew=1.2, label="Censor mark" if i == 0 else "_nolegend_")

        n_fail = int(np.sum(e))
        n_cen = int(len(e) - n_fail)
        comp = f"{n_fail}/8 failed, {n_cen}/8 censored"
        ax.text(0.04, 0.08, comp, transform=ax.transAxes, fontsize=8.5, color=PLOT_COLORS["aux"])
        ax.text(0.02, 0.95, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

        ax.set_title(f"Delta T = {int(s)} deg C")
        ax.set_xlabel("Cycles")
        if i % 3 == 0:
            ax.set_ylabel("Survival probability")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, x_max)
        style_axis(ax)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    handles, labels = dedup_legend(handles, labels)
    add_figure_legend_above(
        fig,
        handles,
        labels,
        ncol=3,
        fontsize=7.6,
        rect_top=0.93,
        legend_y=0.992,
        columnspacing=1.0,
        handlelength=1.9,
    )
    save_figure_dual(
        fig,
        out_dir,
        "main_survival_panels_pub",
        manifest,
        "Main figure",
        "Stress-wise survival comparison using Bayesian PICM-L predictions and Kaplan-Meier with censor marks.",
    )
    plt.close(fig)


def plot_main_reference_case_b10_comparison(
    use_b10_samples: np.ndarray,
    b10_mle: Tuple[float, float, float],
    b10_firth: Tuple[float, float, float],
    b10_prof_unc: Tuple[float, float, float],
    b10_prof_bc: Tuple[float, float, float],
    delta_t_use: float,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main Figure 3: reference-case B10 posterior with method comparison overlays."""
    c_post = PLOT_COLORS["posterior_density"]
    c_bayes = PLOT_COLORS["bayes_fit"]
    c_mle = PLOT_COLORS["mle"]
    c_firth = PLOT_COLORS["firth"]
    c_unc = "#9d4edd"
    c_bc = "#f77f00"

    q025, q50, q975 = np.quantile(use_b10_samples, [0.025, 0.5, 0.975])
    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.9))
    ax.hist(
        use_b10_samples,
        bins=45,
        density=True,
        color=PLOT_COLORS["bayes_band"],
        alpha=0.75,
        edgecolor="white",
        label="Posterior samples",
    )
    if use_b10_samples.size > 40:
        kde = stats.gaussian_kde(use_b10_samples)
        xk = np.linspace(np.min(use_b10_samples), np.max(use_b10_samples), 350)
        ax.plot(xk, kde(xk), color=c_post, lw=2.2, label="Posterior density")

    ax.axvspan(q025, q975, color="#ced4da", alpha=0.35, label="Bayes 95% CrI")
    ax.axvline(q50, color=c_bayes, lw=2.2, label="Bayes median")
    ax.axvline(b10_mle[0], color=c_mle, lw=1.8, ls="--", label="MLE")
    ax.axvline(b10_firth[0], color=c_firth, lw=1.8, ls=(0, (6, 2, 1.8, 2)), label="Firth")

    ymax = ax.get_ylim()[1]
    y1, y2 = 0.88 * ymax, 0.96 * ymax
    ax.hlines(y1, b10_prof_unc[1], b10_prof_unc[2], color=c_unc, lw=3.0, label="Profile CI (uncorrected)")
    ax.hlines(y2, b10_prof_bc[1], b10_prof_bc[2], color=c_bc, lw=3.0, label="Profile CI (Bartlett)")
    ax.plot([b10_prof_unc[0]], [y1], marker="|", color=c_unc, ms=12, mew=2.2)
    ax.plot([b10_prof_bc[0]], [y2], marker="|", color=c_bc, ms=12, mew=2.2)

    annotation = (
        f"Posterior median = {q50:.1f} cycles\n"
        f"95% CrI = [{q025:.1f}, {q975:.1f}]"
    )
    ax.text(
        0.04,
        0.12,
        annotation,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#b5b5b5", alpha=0.94),
    )

    ax.set_xlabel(f"Reference-case B10 at Delta T = {delta_t_use:g} deg C (cycles)")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    add_figure_legend_above(
        fig,
        h,
        l,
        ncol=2,
        fontsize=7.2,
        rect_top=0.76,
        legend_y=0.99,
        columnspacing=1.0,
        handlelength=1.9,
    )
    save_figure_dual(
        fig,
        out_dir,
        "main_reference_case_b10_comparison",
        manifest,
        "Main figure",
        "Reference-case B10 posterior at Delta T = 35 deg C with Bayesian, MLE, Firth, and profile-likelihood interval comparisons.",
    )
    plt.close(fig)

def plot_main_stresswise_life_intervals(
    post_pred: Dict[str, pd.DataFrame],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main Figure 4: forest-style stress-wise median and B10 life intervals."""
    med = post_pred["stress_median"].copy().sort_values("delta_T").reset_index(drop=True)
    b10 = post_pred["stress_b10"].copy().sort_values("delta_T").reset_index(drop=True)
    stress = med["delta_T"].to_numpy(dtype=int)
    ypos = np.arange(stress.size)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.errorbar(
        med["median"].to_numpy(),
        ypos + 0.12,
        xerr=[
            med["median"].to_numpy() - med["eti_2.5%"].to_numpy(),
            med["eti_97.5%"].to_numpy() - med["median"].to_numpy(),
        ],
        fmt="o",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        capsize=3,
        label="Median life (95% CrI)",
    )
    ax.errorbar(
        b10["median"].to_numpy(),
        ypos - 0.12,
        xerr=[
            b10["median"].to_numpy() - b10["eti_2.5%"].to_numpy(),
            b10["eti_97.5%"].to_numpy() - b10["median"].to_numpy(),
        ],
        fmt="s",
        color=PLOT_COLORS["posterior_density"],
        ecolor=PLOT_COLORS["posterior_density"],
        capsize=3,
        label="B10 life (95% CrI)",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"Delta T = {int(s)} deg C" for s in stress])
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Stress level")
    ax.set_title("Stress-wise Predicted Median and B10 Life Intervals")
    style_axis(ax)
    ax.legend(loc="lower right")
    save_figure_dual(
        fig,
        out_dir,
        "main_stresswise_life_intervals",
        manifest,
        "Main figure",
        "Forest-style summary of stress-wise predicted median and B10 life with 95% credible intervals.",
    )
    plt.close(fig)


def plot_main_acceleration_factor(
    bayes_samples: np.ndarray,
    stress_levels: np.ndarray,
    delta_t_use: float,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Main Figure 5: acceleration factor relative to use condition."""
    k = bayes_samples[:, 0]
    b1 = bayes_samples[:, 1]
    b0 = bayes_samples[:, 2]

    x_use = np.log(delta_t_use)
    eta_use = np.exp(b0 - b1 * x_use)
    med_use = weibull_quantile(eta_use, k, 0.5)

    stress = np.array(sorted(stress_levels), dtype=float)
    af_med = []
    af_lo = []
    af_hi = []
    for s in stress:
        eta_s = np.exp(b0 - b1 * np.log(s))
        med_s = weibull_quantile(eta_s, k, 0.5)
        af = med_use / med_s
        ql, qm, qu = np.quantile(af, [0.025, 0.5, 0.975])
        af_lo.append(float(ql))
        af_med.append(float(qm))
        af_hi.append(float(qu))

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.fill_between(stress, af_lo, af_hi, color=PLOT_COLORS["bayes_band"], alpha=0.55, label="95% CrI")
    ax.plot(stress, af_med, color=PLOT_COLORS["bayes_fit"], marker="o", lw=2.1, label="Posterior median AF")
    ax.axhline(1.0, color=PLOT_COLORS["mle"], lw=1.2, ls="--")
    ax.set_xlabel("Delta T (deg C)")
    ax.set_ylabel("Acceleration factor relative to use condition")
    ax.set_title("Acceleration Factor by Stress (Using Median Life)")
    style_axis(ax)
    ax.legend(loc="upper left")
    save_figure_dual(
        fig,
        out_dir,
        "main_acceleration_factor",
        manifest,
        "Main figure",
        "Acceleration factor relative to use condition with posterior median and 95% credible interval.",
    )
    plt.close(fig)


def plot_supp_posterior_pairplot(
    bayes_samples: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: posterior pair/dependence plot for k, beta1, beta0."""
    names = ["k", "beta1", "beta0"]
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 10.2))
    rng = np.random.default_rng(SEED_PRED + 31)
    keep = min(7000, bayes_samples.shape[0])
    samp = bayes_samples[rng.choice(bayes_samples.shape[0], size=keep, replace=False), :]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            xi = samp[:, j]
            yi = samp[:, i]
            if i == j:
                ax.hist(xi, bins=35, density=True, color=PLOT_COLORS["bayes_band"], alpha=0.75, edgecolor="white")
                kde = stats.gaussian_kde(xi)
                xx = np.linspace(np.min(xi), np.max(xi), 250)
                ax.plot(xx, kde(xx), color=PLOT_COLORS["posterior_density"], lw=1.9)
                ax.set_ylabel("Density")
            else:
                ax.scatter(xi, yi, s=6, alpha=0.18, color=PLOT_COLORS["bayes_fit"], edgecolors="none")
                r = np.corrcoef(xi, yi)[0, 1]
                ax.text(0.05, 0.9, f"r={r:.2f}", transform=ax.transAxes, fontsize=8.5, color=PLOT_COLORS["aux"])
            if i == 2:
                ax.set_xlabel(names[j])
            if j == 0:
                ax.set_ylabel(names[i] if i != j else "Density")
            style_axis(ax)
    fig.suptitle("Posterior Pair Plot for PICM-L Parameters", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure_dual(
        fig,
        out_dir,
        "supp_posterior_pairplot",
        manifest,
        "Supplementary",
        "Pairwise posterior dependence for PICM-L parameters (k, beta1, beta0).",
    )
    plt.close(fig)


def plot_supp_censoring_burden(
    data: ReliabilityData,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: censoring burden by stress."""
    tmp = (
        data.df.groupby("delta_T", as_index=False)
        .agg(n_failed=("event", lambda s: int(np.sum(s == 1))), n_censored=("event", lambda s: int(np.sum(s == 0))))
        .sort_values("delta_T")
    )
    x = tmp["delta_T"].to_numpy(dtype=int)
    fail = tmp["n_failed"].to_numpy(dtype=int)
    cen = tmp["n_censored"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars1 = ax.bar(x, fail, color=PLOT_COLORS["bayes_fit"], label="Failed", width=9)
    bars2 = ax.bar(x, cen, bottom=fail, color="#adb5bd", label="Right-censored", width=9)
    for b1, b2, f, c, xx in zip(bars1, bars2, fail, cen, x):
        y_fail = f / 2 + 0.05 if f > 0 else 0.18
        txt_color = "white" if f > 0 else PLOT_COLORS["aux"]
        ax.text(b1.get_x() + b1.get_width() / 2, y_fail, f"{f}", ha="center", va="center", color=txt_color, fontsize=9)
        if c > 0:
            ax.text(
                b2.get_x() + b2.get_width() / 2,
                f + c / 2,
                f"{c}",
                ha="center",
                va="center",
                color=PLOT_COLORS["aux"],
                fontsize=9,
            )
        if f == 0:
            ax.text(float(xx), f + c + 0.18, "0 failures", ha="center", va="bottom", fontsize=8.5, color="#8a2b2b")

    ax.text(0.01, 0.98, f"Total sample size: N={data.n}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.set_xlabel("Delta T (deg C)")
    ax.set_ylabel("Unit count")
    ax.set_title("Censoring Burden by Stress Level")
    ax.set_xticks(x)
    ax.set_ylim(0, max(8.6, float(np.max(fail + cen) + 0.8)))
    style_axis(ax)
    ax.legend(loc="upper right")
    save_figure_dual(
        fig,
        out_dir,
        "supp_censoring_burden",
        manifest,
        "Supplementary",
        "Stacked bar chart of failures and right-censored units by stress level, with zero-failure annotation and total sample size.",
    )
    plt.close(fig)

def plot_supp_cdf_panels(
    data: ReliabilityData,
    bayes_samples: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: stress-wise cumulative-failure panels."""
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(SEED_PRED + 32)
    subs = bayes_samples[rng.choice(bayes_samples.shape[0], size=min(5000, bayes_samples.shape[0]), replace=False), :]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, s in enumerate(data.stress_levels):
        ax = axes[i]
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        e = data.event[idx]
        x = np.log(s)
        x_max = max(1.15 * np.max(y), 1.0)
        yg = np.linspace(1e-3, x_max, 250)

        k_sub = subs[:, 0][:, None]
        eta_sub = np.exp(subs[:, 2][:, None] - subs[:, 1][:, None] * x)
        surv = np.exp(-((yg[None, :] / eta_sub) ** k_sub))
        cdf = 1.0 - surv
        ql, qm, qu = np.quantile(cdf, [0.025, 0.5, 0.975], axis=0)
        ax.fill_between(
            yg,
            ql,
            qu,
            color=PLOT_COLORS["bayes_band"],
            alpha=0.58,
            label="95% predictive band" if i == 0 else "_nolegend_",
        )
        ax.plot(yg, qm, color=PLOT_COLORS["bayes_fit"], lw=2.0, label="Posterior predictive median CDF" if i == 0 else "_nolegend_")

        ft, fs, _ct = km_curve(y, e)
        if ft.size > 0:
            ax.step(
                np.r_[0.0, ft],
                np.r_[0.0, 1.0 - fs],
                where="post",
                color=PLOT_COLORS["km_empirical"],
                lw=1.7,
                label="Empirical CDF (KM)" if i == 0 else "_nolegend_",
            )
        ax.text(0.02, 0.95, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_title(f"Delta T = {int(s)} deg C")
        ax.set_xlabel("Cycles")
        if i % 3 == 0:
            ax.set_ylabel("Cumulative failure probability")
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.02)
        style_axis(ax)

    handles, labels = dedup_legend(
        [h for ax in axes for h in ax.get_legend_handles_labels()[0]],
        [l for ax in axes for l in ax.get_legend_handles_labels()[1]],
    )
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Posterior Predictive CDF Panels by Stress", y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_figure_dual(
        fig,
        out_dir,
        "supp_cdf_panels",
        manifest,
        "Supplementary",
        "Stress-wise cumulative-failure comparison between empirical curves and posterior predictive CDF bands.",
    )
    plt.close(fig)


def plot_supp_weibull_residual_diagnostics(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: Cox-Snell and quantile diagnostics for Weibull PICM-L."""
    k, b1, b0 = [float(v) for v in mle_theta]
    eta = np.exp(b0 - b1 * data.x_log)
    resid = (data.y / eta) ** k
    ev = data.event.astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # Cox-Snell residual check: cumulative hazard vs residual should follow y=x.
    ft, fs, _ct = km_curve(resid, ev)
    if ft.size > 0:
        haz = -np.log(np.clip(fs, 1e-12, 1.0))
        axes[0].step(
            np.r_[0.0, ft],
            np.r_[0.0, haz],
            where="post",
            color=PLOT_COLORS["bayes_fit"],
            lw=1.9,
            label="Estimated cumulative hazard",
        )
    rline = np.linspace(0.0, max(4.0, np.max(resid) * 1.05), 200)
    axes[0].plot(rline, rline, color=PLOT_COLORS["mle"], ls="--", lw=1.5, label="Reference y=x")
    finite_haz = haz[np.isfinite(haz)] if ft.size > 0 else np.array([4.0])
    ycap = max(4.0, float(np.quantile(finite_haz, 0.95)) * 1.15)
    axes[0].set_ylim(0.0, ycap)
    axes[0].set_xlabel("Cox-Snell residual")
    axes[0].set_ylabel("Cumulative hazard")
    axes[0].set_title("Cox-Snell Residual Diagnostic (axis truncated)")
    style_axis(axes[0])
    axes[0].legend(loc="upper left")

    # Failed-unit quantile plot vs Exp(1) quantiles.
    rf = np.sort(resid[ev == 1])
    if rf.size >= 2:
        p = (np.arange(1, rf.size + 1) - 0.5) / rf.size
        tq = stats.expon.ppf(p)
        axes[1].scatter(tq, rf, s=24, color=PLOT_COLORS["posterior_density"], alpha=0.85, label="Observed residual quantiles")
        lim = max(float(np.max(tq)), float(np.max(rf))) * 1.05
        axes[1].plot([0, lim], [0, lim], color=PLOT_COLORS["mle"], ls="--", lw=1.5, label="Reference y=x")
    axes[1].set_xlabel("Theoretical Exp(1) quantiles")
    axes[1].set_ylabel("Observed failed-unit residual quantiles")
    axes[1].set_title("Residual Quantile Plot")
    style_axis(axes[1])
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    save_figure_dual(
        fig,
        out_dir,
        "supp_weibull_residual_diagnostics",
        manifest,
        "Supplementary",
        "Weibull residual diagnostics using Cox-Snell cumulative hazard and residual quantile comparisons.",
    )
    plt.close(fig)


def plot_supp_leave_one_stress_out(
    data: ReliabilityData,
    out_dir: Path,
    manifest: List[Dict[str, str]],
    enable: bool = True,
    loso_table: Optional[pd.DataFrame] = None,
) -> None:
    """Supplementary: leave-one-stress-out predictive check (observed-vs-predicted with uncertainty bars)."""
    if not enable:
        warnings.warn("Leave-one-stress-out plot disabled by flag.")
        return

    if loso_table is None:
        chk = compute_loso_predictive_adequacy_picm_l(
            data=data,
            prior_cfg=get_prior_config(BASELINE_PRIOR_KEY),
            max_stress=LOSO_MAX_STRESS,
        )
    else:
        chk = loso_table.copy()

    if chk.empty:
        warnings.warn("LOSO table is empty; skipping LOSO figure.")
        return

    chk = chk.sort_values("held_out_stress").reset_index(drop=True)
    x = chk["observed_failure_fraction"].to_numpy(dtype=float)
    y = chk["predicted_failure_fraction"].to_numpy(dtype=float)
    ylo = chk["predicted_pi_2_5"].to_numpy(dtype=float)
    yhi = chk["predicted_pi_97_5"].to_numpy(dtype=float)
    stress = chk["held_out_stress"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDE, 2.95))
    ax.errorbar(
        x,
        y,
        yerr=[y - ylo, yhi - y],
        fmt="o",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        capsize=3,
        lw=1.4,
        ms=6,
        label="LOSO prediction (95% PI)",
    )
    ax.plot([0, 1], [0, 1], color=PLOT_COLORS["mle"], ls="--", lw=1.3, label="45-degree reference")

    label_counts: Dict[Tuple[float, float], int] = {}
    for xx, yy, ss, rr in zip(x, y, stress, chk.itertuples(index=False)):
        txt = f"{int(ss)} deg C ({int(rr.observed_failed)}/{int(rr.observed_total)})"
        key = (round(float(xx), 3), round(float(yy), 3))
        duplicate_index = label_counts.get(key, 0)
        label_counts[key] = duplicate_index + 1

        x_txt = float(xx) + 0.014
        y_txt = float(yy) + 0.016 + 0.022 * duplicate_index
        ha = "left"
        va = "bottom"
        if float(xx) <= 0.02 and float(yy) <= 0.02:
            x_txt = 0.06
            y_txt = 0.04 + 0.045 * duplicate_index
        if float(xx) >= 0.95:
            x_txt = 0.93
            ha = "right"
        if float(yy) >= 0.96:
            y_txt = 0.98 - 0.10 * duplicate_index
            va = "top"
        elif float(xx) >= 0.95 and float(yy) >= 0.70:
            y_txt = float(yy) + 0.03

        ax.text(
            x_txt,
            y_txt,
            txt,
            fontsize=7.6,
            color=PLOT_COLORS["aux"],
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72),
        )

    ax.set_xlabel("Observed failure fraction (held-out stress)")
    ax.set_ylabel("Predicted failure fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)
    ax.legend(loc="upper left", fontsize=7.4)
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.985], pad=0.6)
    save_figure_dual(
        fig,
        out_dir,
        "supp_leave_one_stress_out",
        manifest,
        "Supplementary",
        "Leave-one-stress-out observed-vs-predicted failure fractions with 45-degree reference and predictive uncertainty bars.",
    )
    plt.close(fig)

def plot_supp_prior_vs_posterior_refined(
    bayes_samples: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: refined prior-vs-posterior overlays."""
    names = ["k", "beta1", "beta0"]
    cfg = get_prior_config(BASELINE_PRIOR_KEY)
    sf_k = float(stats.gamma.sf(cfg["k_trunc_lower"], a=cfg["k_gamma_shape"], scale=1.0 / cfg["k_gamma_rate"]))
    if sf_k <= 0.0:
        sf_k = 1.0
    prior_pdf = [
        lambda x: np.where(
            np.asarray(x, dtype=float) > cfg["k_trunc_lower"],
            stats.gamma.pdf(x, a=cfg["k_gamma_shape"], scale=1.0 / cfg["k_gamma_rate"]) / sf_k,
            0.0,
        ),
        lambda x: stats.truncnorm.pdf(
            x,
            (cfg["beta1_trunc_lower"] - cfg["beta1_mean"]) / cfg["beta1_sd"],
            np.inf,
            loc=cfg["beta1_mean"],
            scale=cfg["beta1_sd"],
        ),
        lambda x: stats.norm.pdf(x, loc=cfg["beta0_mean"], scale=cfg["beta0_sd"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    for j, ax in enumerate(axes):
        x = bayes_samples[:, j]
        lo, hi = np.quantile(x, [0.005, 0.995])
        xx = np.linspace(lo, hi, 350)
        ax.hist(x, bins=40, density=True, color=PLOT_COLORS["bayes_band"], alpha=0.75, edgecolor="white", label="Posterior")
        kde = stats.gaussian_kde(x)
        ax.plot(xx, kde(xx), color=PLOT_COLORS["posterior_density"], lw=2.0, label="Posterior KDE")
        ax.plot(xx, prior_pdf[j](xx), color=PLOT_COLORS["km_empirical"], lw=1.3, label="Prior")
        ax.axvline(float(np.mean(x)), color=PLOT_COLORS["bayes_fit"], ls="--", lw=1.2, label="Posterior mean")
        ax.axvline(float(np.median(x)), color=PLOT_COLORS["bayes_fit"], ls=":", lw=1.2, label="Posterior median")
        ax.set_xlabel(names[j])
        ax.set_ylabel("Density")
        ax.set_title(f"{names[j]}: prior vs posterior")
        style_axis(ax)
        h, l = dedup_legend(*ax.get_legend_handles_labels())
        ax.legend(h, l, loc="upper right", fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    fig.tight_layout()
    save_figure_dual(
        fig,
        out_dir,
        "supp_prior_vs_posterior_refined",
        manifest,
        "Supplementary",
        "Refined prior-vs-posterior overlays with posterior mean and median annotations.",
    )
    plt.close(fig)


def plot_supp_profile_likelihood_refined(
    profile_res: Dict[str, object],
    bartlett_factor: float,
    b10_prof_unc: Tuple[float, float, float],
    b10_prof_bc: Tuple[float, float, float],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: refined profile likelihood figure for reference-case B10."""
    lg = profile_res["logpsi_grid"]
    b10_grid = np.exp(lg)
    lr_unc = profile_res["lr_grid"]
    lr_bc = lr_unc / bartlett_factor
    q95 = profile_res["q95"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(b10_grid, lr_unc, color=PLOT_COLORS["bayes_fit"], lw=2.2, label="Uncorrected LR profile")
    ax.plot(b10_grid, lr_bc, color=PLOT_COLORS["km_empirical"], lw=2.0, ls="--", label="Bartlett-corrected LR profile")
    ax.axhline(q95, color=PLOT_COLORS["mle"], lw=1.5, ls=":", label="Chi-square threshold (95%, df=1)")

    ax.axvline(b10_prof_unc[0], color=PLOT_COLORS["bayes_fit"], lw=1.1, alpha=0.8)
    ax.axvline(b10_prof_unc[1], color="#9d4edd", lw=1.1, ls="--", label="Uncorrected CI endpoints")
    ax.axvline(b10_prof_unc[2], color="#9d4edd", lw=1.1, ls="--")
    ax.axvline(b10_prof_bc[1], color="#f77f00", lw=1.1, ls="-.", label="Bartlett CI endpoints")
    ax.axvline(b10_prof_bc[2], color="#f77f00", lw=1.1, ls="-.")

    ax.set_xlabel("Reference-case B10 (cycles)")
    ax.set_ylabel("Likelihood-ratio statistic")
    ax.set_title("Refined Profile Likelihood for Reference-Case B10")
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    ax.legend(h, l, loc="upper right")
    save_figure_dual(
        fig,
        out_dir,
        "supp_profile_likelihood_b10_refined",
        manifest,
        "Supplementary",
        "Refined profile-likelihood plot with uncorrected and Bartlett-corrected profiles and CI endpoints.",
    )
    plt.close(fig)


def plot_supp_trace_plots_refined(
    chains: np.ndarray,
    rhat: np.ndarray,
    ess: np.ndarray,
    accept_rates: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: refined MCMC trace plots with diagnostics annotations."""
    names = ["k", "beta1", "beta0"]
    colors = [PLOT_COLORS["bayes_fit"], PLOT_COLORS["chain_2"], PLOT_COLORS["km_empirical"]]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True)
    for j, ax in enumerate(axes):
        for c in range(chains.shape[0]):
            ax.plot(chains[c, :, j], lw=0.55, alpha=0.75, color=colors[c % len(colors)], label=f"Chain {c+1}" if j == 0 else None)
        ax.set_ylabel(names[j])
        ax.set_title(f"{names[j]}  |  R-hat={rhat[j]:.3f}, ESS={ess[j]:.0f}", loc="left", fontsize=10)
        style_axis(ax)
    axes[-1].set_xlabel("Post-burn iteration")
    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle(
        "Refined Trace Plots (acceptance: "
        + ", ".join([f"chain{c+1}={accept_rates[c]:.3f}" for c in range(len(accept_rates))])
        + ")",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_figure_dual(
        fig,
        out_dir,
        "supp_trace_plots_refined",
        manifest,
        "Supplementary",
        "Refined trace plots with R-hat, ESS, and chain acceptance-rate annotations.",
    )
    plt.close(fig)


def plot_supp_failure_fraction_refined(
    failure_fraction_check: pd.DataFrame,
    data: ReliabilityData,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: refined observed-vs-predictive failure-fraction check."""
    ff = failure_fraction_check.copy().sort_values("delta_T").reset_index(drop=True)
    stress = ff["delta_T"].to_numpy(dtype=int)
    obs = ff["observed_failure_fraction"].to_numpy(dtype=float)
    pred = ff["predicted_median_failure_fraction"].to_numpy(dtype=float)
    lo = ff["predicted_pi_2.5%"].to_numpy(dtype=float)
    hi = ff["predicted_pi_97.5%"].to_numpy(dtype=float)
    counts = data.df.groupby("delta_T")["event"].sum().reindex(stress).fillna(0).to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(8.9, 5.3))
    ax.fill_between(stress, lo, hi, color=PLOT_COLORS["bayes_band"], alpha=0.55, label="Posterior predictive 95% PI")
    ax.plot(stress, pred, marker="s", color=PLOT_COLORS["bayes_fit"], lw=2.0, label="Posterior predictive median")
    ax.plot(stress, obs, marker="o", color=PLOT_COLORS["failed"], lw=1.7, label="Observed failure fraction")
    for x, y, c in zip(stress, obs, counts):
        ax.text(x, min(y + 0.04, 0.98), f"{c}/8", ha="center", va="bottom", fontsize=8.5, color=PLOT_COLORS["aux"])

    ax.set_xlabel("Delta T (deg C)")
    ax.set_ylabel("Failure fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(stress)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_title("Refined Failure-Fraction Posterior Predictive Check")
    style_axis(ax)
    h, l = dedup_legend(*ax.get_legend_handles_labels())
    ax.legend(h, l, loc="lower right")
    save_figure_dual(
        fig,
        out_dir,
        "supp_failure_fraction_check_refined",
        manifest,
        "Supplementary",
        "Observed failure fractions with posterior predictive median and 95% predictive interval by stress.",
    )
    plt.close(fig)


def plot_supp_weibull_probability_plots_refined(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Supplementary: refined Weibull probability plots by stress, with low-information panels greyed out."""
    k_mle, b1_mle, b0_mle = [float(v) for v in mle_theta]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, s in enumerate(data.stress_levels):
        ax = axes[i]
        idx = data.stress_to_idx[float(s)]
        y = data.y[idx]
        e = data.event[idx]
        ft, fs, _ct = km_curve(y, e)
        n_fail = int(np.sum(e))
        eta_mle = np.exp(b0_mle - b1_mle * np.log(s))
        xline = np.linspace(np.log(max(np.min(y) * 0.9, 1e-3)), np.log(np.max(y) * 1.1), 150)
        yline = k_mle * (xline - np.log(eta_mle))
        ax.plot(xline, yline, color=PLOT_COLORS["bayes_fit"], lw=2.0, label=f"Fitted line (k={k_mle:.2f})")
        if n_fail >= 2 and ft.size > 1:
            mask = (fs > 0.0) & (fs < 1.0)
            xx = np.log(ft[mask])
            yy = np.log(-np.log(fs[mask]))
            ax.scatter(xx, yy, c=PLOT_COLORS["failed"], s=28, alpha=0.9, label="KM points")
        else:
            ax.axhspan(*ax.get_ylim(), color="#f3f4f6", alpha=0.85, zorder=-1)
            ax.text(0.5, 0.5, "All units censored;\nprobability plot not informative", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, color="#4b5563")
        ax.text(0.02, 0.95, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_title(f"Delta T = {int(s)} deg C")
        if i % 3 == 0:
            ax.set_ylabel("log(-log(S))")
        if i >= 3:
            ax.set_xlabel("log(Cycles)")
        style_axis(ax)

    h, l = dedup_legend(
        [h for ax in axes for h in ax.get_legend_handles_labels()[0]],
        [l for ax in axes for l in ax.get_legend_handles_labels()[1]],
    )
    fig.legend(h, l, loc="lower center", ncol=2)
    fig.suptitle("Refined Weibull Probability Plots by Stress", y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    for ax in axes:
        ax.tick_params(labelsize=MIN_PANEL_TICK_PT)
    save_figure_dual(
        fig,
        out_dir,
        "supp_weibull_probability_plots_refined",
        manifest,
        "Supplementary",
        "Weibull probability plots by stress with low-information fully censored panels marked as not informative.",
    )
    plt.close(fig)


def generate_publication_figures(
    data: ReliabilityData,
    mle_theta: np.ndarray,
    firth_theta: np.ndarray,
    picm_c_theta: np.ndarray,
    bayes_samples: np.ndarray,
    pred_picm_l_grid: Dict[str, object],
    pred_picm_c_grid: Dict[str, object],
    pred_lognorm_grid: Dict[str, object],
    pred_picm_l_high_grid: Dict[str, object],
    pred_picm_l_targets: Dict[str, object],
    pred_picm_c_targets: Dict[str, object],
    pred_lognorm_targets: Dict[str, object],
    table_high_dt_posterior: pd.DataFrame,
    chains: np.ndarray,
    rhat: np.ndarray,
    ess: np.ndarray,
    accept_rates: np.ndarray,
    post_pred: Dict[str, pd.DataFrame],
    profile_res: Dict[str, object],
    bartlett_factor: float,
    b10_mle: Tuple[float, float, float],
    b10_firth: Tuple[float, float, float],
    b10_prof_unc: Tuple[float, float, float],
    b10_prof_bc: Tuple[float, float, float],
    failure_fraction_check: pd.DataFrame,
    loso_table: pd.DataFrame,
    prior_sensitivity_b10_35: Dict[str, np.ndarray],
    delta_t_use: float,
    main_dir: Path,
    supp_dir: Path,
    manifest_path: Path,
) -> None:
    """Generate main and supplementary publication figures with graceful failure handling."""
    ALL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, str]] = []
    jobs = [
        (
            "main stress-life regime",
            plot_main_stress_life_logy,
            (data, pred_picm_l_grid, pred_picm_c_grid, pred_lognorm_grid, main_dir, manifest),
        ),
        (
            "main failure-fraction calibration",
            plot_main_failure_fraction_calibration,
            (failure_fraction_check, main_dir, manifest),
        ),
        (
            "main reference-case B10 comparison",
            plot_main_reference_case_b10_comparison,
            (
                post_pred["use_b10_samples"],
                b10_mle,
                b10_firth,
                b10_prof_unc,
                b10_prof_bc,
                delta_t_use,
                main_dir,
                manifest,
            ),
        ),
        (
            "main b10-vs-deltaT",
            plot_main_b10_vs_delta_t,
            (B10_GRID_DELTA_T, pred_picm_l_grid, pred_picm_c_grid, pred_lognorm_grid, main_dir, manifest, MISSION_CYCLE_TARGETS),
        ),
        ("supp prior sensitivity B10@35C", plot_prior_sensitivity_b10_35c, (prior_sensitivity_b10_35, supp_dir, manifest)),
        ("supp prior-vs-posterior", plot_supp_prior_vs_posterior_refined, (bayes_samples, supp_dir, manifest)),
        (
            "supp profile refined",
            plot_supp_profile_likelihood_refined,
            (profile_res, bartlett_factor, b10_prof_unc, b10_prof_bc, supp_dir, manifest),
        ),
        ("supp trace refined", plot_supp_trace_plots_refined, (chains, rhat, ess, accept_rates, supp_dir, manifest)),
        ("supp survival panels", plot_main_survival_panels, (data, bayes_samples, mle_theta, supp_dir, manifest)),
        ("supp residual diagnostics", plot_supp_weibull_residual_diagnostics, (data, mle_theta, supp_dir, manifest)),
        ("supp leave-one-stress-out", plot_supp_leave_one_stress_out, (data, supp_dir, manifest, ENABLE_LEAVE_ONE_STRESS_OUT, loso_table)),
        ("supp Weibull probability refined", plot_supp_weibull_probability_plots_refined, (data, mle_theta, supp_dir, manifest)),
        (
            "supp B10 density 10C-15C",
            plot_main_b10_density_10c_15c,
            (pred_picm_l_targets, pred_picm_c_targets, pred_lognorm_targets, supp_dir, manifest),
        ),
        (
            "supp high-deltaT extrapolation CI",
            plot_high_deltaT_extrapolation_ci,
            (pred_picm_l_high_grid, table_high_dt_posterior, supp_dir, manifest),
        ),
    ]

    for name, fn, args in jobs:
        try:
            fn(*args)
        except Exception as exc:
            warnings.warn(f"Skipped figure '{name}' due to error: {exc}")


# =============================================================================
# Tables and summary outputs
# =============================================================================



def build_table_loso_summary(loso_table: pd.DataFrame) -> pd.DataFrame:
    """Compact LOSO adequacy table intended for reporting."""
    if loso_table is None or loso_table.empty:
        return pd.DataFrame(columns=[
            "Held-out Delta T (deg C)", "Observed fraction", "Predicted median", "95% PI", "Absolute error", "Coverage"
        ])
    chk = loso_table.sort_values("held_out_stress").reset_index(drop=True).copy()
    return pd.DataFrame({
        "Held-out Delta T (deg C)": chk["held_out_stress"].astype(float),
        "Observed fraction": chk["observed_failure_fraction"].astype(float),
        "Predicted median": chk["predicted_failure_fraction"].astype(float),
        "95% PI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(chk["predicted_pi_2_5"], chk["predicted_pi_97_5"])],
        "Absolute error": chk["absolute_error"].astype(float),
        "Coverage": chk["coverage_95ppi"].astype(bool),
    })


def write_loso_summary_outputs(loso_table: pd.DataFrame, out_dir: Path) -> None:
    """Write compact LOSO adequacy outputs for submission assembly."""
    tab = build_table_loso_summary(loso_table)
    tab.to_csv(out_dir / "table_loso_reference_summary.csv", index=False)
    if tab.empty:
        return
    latex_df = tab.copy()
    latex_df["Observed fraction"] = latex_df["Observed fraction"].map(lambda v: f"{v:.3f}")
    latex_df["Predicted median"] = latex_df["Predicted median"].map(lambda v: f"{v:.3f}")
    latex_df["Absolute error"] = latex_df["Absolute error"].map(lambda v: f"{v:.3f}")
    latex_df["Coverage"] = latex_df["Coverage"].map(lambda v: "Yes" if bool(v) else "No")
    latex = latex_df.to_latex(index=False, escape=False)
    (out_dir / "table_loso_reference_summary.tex").write_text(latex, encoding="utf-8")


def build_table_b(
    mle: Dict[str, object],
    firth: Dict[str, object],
    post_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Table B: PICM-L parameter estimates."""
    params = ["k", "beta1", "beta0"]
    rows = []
    for i, p in enumerate(params):
        ps = post_summary.loc[post_summary["parameter"] == p].iloc[0]
        rows.append(
            {
                "parameter": p,
                "MLE": float(mle["theta"][i]),
                "MLE_SE": float(mle["se"][i]) if np.isfinite(mle["se"][i]) else np.nan,
                "Firth": float(firth["theta"][i]),
                "Firth_SE": float(firth["se_pen"][i]) if np.isfinite(firth["se_pen"][i]) else np.nan,
                "Firth_role": "point-estimate consistency check; covariance ridge-stabilised, not used for inference",
                "Firth_ridge_trigger_count": int(firth.get("ridge_trigger_count", len(firth.get("ridge_events", [])))),
                "Firth_rejected_hessian_count": int(firth.get("rejected_hessian_count", firth.get("non_pd_count", 0))),
                "Bayes_mean": float(ps["mean"]),
                "Bayes_median": float(ps["median"]),
                "Bayes_ETI_2.5%": float(ps["eti_2.5%"]),
                "Bayes_ETI_97.5%": float(ps["eti_97.5%"]),
                "Bayes_HPD_2.5%": float(ps["hpd_2.5%"]),
                "Bayes_HPD_97.5%": float(ps["hpd_97.5%"]),
            }
        )
    return pd.DataFrame(rows)


def build_table_c(
    picm_l_mle: Dict[str, object],
    picm_c: Dict[str, object],
    lognorm: Dict[str, object],
    bayes_ic: Dict[str, float],
    n_obs: int,
) -> pd.DataFrame:
    """Table C: model comparison."""
    ll_l = float(picm_l_mle["loglik"])
    aic_l, bic_l = calc_aic_bic(ll_l, n_params=3, n_obs=n_obs)
    return pd.DataFrame(
        [
            {
                "model": "Weibull PICM-L (primary)",
                "Uncertainty method": "MCMC posterior sampling",
                "logLik": ll_l,
                "n_params": 3,
                "AIC": aic_l,
                "BIC": bic_l,
                "WAIC_PICM_L_only": bayes_ic["waic"],
                "p_WAIC_PICM_L_only": bayes_ic["p_waic"],
                "DIC_PICM_L_supplementary": bayes_ic["dic"],
                "Criteria comparability note": "AIC/BIC comparable across listed models; WAIC/DIC reported for PICM-L only.",
            },
            {
                "model": "Weibull PICM-C (sensitivity)",
                "Uncertainty method": "Penalized fit + Laplace approximation",
                "logLik": float(picm_c["loglik"]),
                "n_params": 4,
                "AIC": float(picm_c["aic"]),
                "BIC": float(picm_c["bic"]),
                "WAIC_PICM_L_only": np.nan,
                "p_WAIC_PICM_L_only": np.nan,
                "DIC_PICM_L_supplementary": np.nan,
                "Criteria comparability note": "AIC/BIC comparable across listed models; WAIC/DIC reported for PICM-L only.",
            },
            {
                "model": "Lognormal AFT (sensitivity)",
                "Uncertainty method": "MLE fit + Laplace approximation",
                "logLik": float(lognorm["loglik"]),
                "n_params": 3,
                "AIC": float(lognorm["aic"]),
                "BIC": float(lognorm["bic"]),
                "WAIC_PICM_L_only": np.nan,
                "p_WAIC_PICM_L_only": np.nan,
                "DIC_PICM_L_supplementary": np.nan,
                "Criteria comparability note": "AIC/BIC comparable across listed models; WAIC/DIC reported for PICM-L only.",
            },
        ]
    )


def build_table_d(
    delta_t_use: float,
    b10_mle: Tuple[float, float, float],
    b10_firth: Tuple[float, float, float],
    b10_bayes: Tuple[float, float, float],
    b10_prof_unc: Tuple[float, float, float],
    b10_prof_bc: Tuple[float, float, float],
    firth: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Table D: reference-case reliability (B10)."""
    firth_ridge_count = int(firth.get("ridge_trigger_count", len(firth.get("ridge_events", [])))) if firth else 0
    firth_rejected_count = int(firth.get("rejected_hessian_count", firth.get("non_pd_count", 0))) if firth else 0
    return pd.DataFrame(
        [
            {
                "method": "Bayesian posterior (ETI)",
                "delta_T_use": delta_t_use,
                "B10_cycles_estimate": b10_bayes[0],
                "lower_95": b10_bayes[1],
                "upper_95": b10_bayes[2],
                "interval_priority": 1,
                "interval_note": "Primary uncertainty interval.",
                "Firth_ridge_trigger_count": np.nan,
                "Firth_rejected_hessian_count": np.nan,
            },
            {
                "method": "Profile likelihood (Bartlett-corrected)",
                "delta_T_use": delta_t_use,
                "B10_cycles_estimate": b10_prof_bc[0],
                "lower_95": b10_prof_bc[1],
                "upper_95": b10_prof_bc[2],
                "interval_priority": 2,
                "interval_note": "Primary frequentist uncertainty interval.",
                "Firth_ridge_trigger_count": np.nan,
                "Firth_rejected_hessian_count": np.nan,
            },
            {
                "method": "MLE delta-method",
                "delta_T_use": delta_t_use,
                "B10_cycles_estimate": b10_mle[0],
                "lower_95": b10_mle[1],
                "upper_95": b10_mle[2],
                "interval_priority": 3,
                "interval_note": "Asymptotic reference interval.",
                "Firth_ridge_trigger_count": np.nan,
                "Firth_rejected_hessian_count": np.nan,
            },
            {
                "method": "Profile likelihood (uncorrected)",
                "delta_T_use": delta_t_use,
                "B10_cycles_estimate": b10_prof_unc[0],
                "lower_95": b10_prof_unc[1],
                "upper_95": b10_prof_unc[2],
                "interval_priority": 4,
                "interval_note": "Diagnostic profile interval; Bartlett-corrected profile is preferred.",
                "Firth_ridge_trigger_count": np.nan,
                "Firth_rejected_hessian_count": np.nan,
            },
            {
                "method": "Firth point estimate (consistency check)",
                "delta_T_use": delta_t_use,
                "B10_cycles_estimate": b10_firth[0],
                "lower_95": np.nan,
                "upper_95": np.nan,
                "interval_priority": 5,
                "interval_note": "Diagnostic only; covariance ridge-stabilised, not used for inference.",
                "Firth_ridge_trigger_count": firth_ridge_count,
                "Firth_rejected_hessian_count": firth_rejected_count,
            },
        ]
    )


def b10_delta_interval(theta: np.ndarray, cov: Optional[np.ndarray], delta_t_use: float) -> Tuple[float, float, float]:
    """
    Delta-method interval for B10 at use condition.
    Returns (estimate, lower, upper).
    """
    psi = b10_use_from_theta(theta, delta_t_use)
    if cov is None:
        return psi, np.nan, np.nan

    k, b1, _b0 = theta
    x_use = np.log(delta_t_use)
    c = -np.log(0.9)
    grad = np.array(
        [
            psi * (-np.log(c) / (k**2)),
            psi * (-x_use),
            psi,
        ],
        dtype=float,
    )
    var = float(grad @ cov @ grad)
    if var <= 0 or not np.isfinite(var):
        return psi, np.nan, np.nan
    se = float(np.sqrt(var))
    lo = max(psi - 1.96 * se, 1e-12)
    hi = psi + 1.96 * se
    return float(psi), float(lo), float(hi)


def build_result_values_payload(
    delta_t_use: float,
    waic_dic: Dict[str, float],
    table_c: pd.DataFrame,
    table_d: pd.DataFrame,
    loso_table: pd.DataFrame,
    prior_sensitivity_table: pd.DataFrame,
    modelwise_table: pd.DataFrame,
) -> Dict[str, object]:
    """Build a compact machine-readable result-value payload."""
    def _safe_records(df: pd.DataFrame) -> List[Dict[str, object]]:
        records = df.to_dict(orient="records")
        out: List[Dict[str, object]] = []
        for rec in records:
            clean: Dict[str, object] = {}
            for k, v in rec.items():
                if isinstance(v, (float, np.floating)) and (not np.isfinite(v)):
                    clean[k] = None
                else:
                    clean[k] = v
            out.append(clean)
        return out

    payload: Dict[str, object] = {}
    payload["delta_t_use_degC"] = float(delta_t_use)
    payload["waic_picm_l_only"] = float(waic_dic["waic"])
    payload["p_waic_picm_l_only"] = float(waic_dic["p_waic"])
    payload["dic_picm_l_supplementary"] = float(waic_dic["dic"])
    payload["model_comparison_note"] = "WAIC/DIC are PICM-L-only; AIC/BIC are cross-model comparable."

    payload["table_d_use_condition"] = _safe_records(table_d)
    payload["table_c_model_comparison"] = _safe_records(table_c)

    if not loso_table.empty:
        payload["loso_mean_absolute_error"] = float(np.mean(loso_table["absolute_error"]))
        payload["loso_coverage_rate_95ppi"] = float(np.mean(loso_table["coverage_95ppi"].astype(float)))
        payload["loso_rows"] = _safe_records(loso_table)

    if not prior_sensitivity_table.empty:
        payload["prior_sensitivity_rows"] = _safe_records(prior_sensitivity_table)

    target_35 = modelwise_table[np.isclose(modelwise_table["delta_T"], 35.0)].copy()
    if not target_35.empty:
        payload["b10_at_35_by_model"] = _safe_records(target_35[["model", "b10_q50", "b10_q025", "b10_q975"]])
    return payload


def write_result_values_files(payload: Dict[str, object], txt_path: Path, json_path: Path) -> None:
    """Write selected result values in human-readable and machine-readable formats."""
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "Manuscript Values",
        "=================",
        f"delta_t_use_degC: {payload.get('delta_t_use_degC')}",
        f"waic_picm_l_only: {payload.get('waic_picm_l_only')}",
        f"p_waic_picm_l_only: {payload.get('p_waic_picm_l_only')}",
        f"dic_picm_l_supplementary: {payload.get('dic_picm_l_supplementary')}",
        f"model_comparison_note: {payload.get('model_comparison_note')}",
    ]
    if "loso_mean_absolute_error" in payload:
        lines.append(f"loso_mean_absolute_error: {payload['loso_mean_absolute_error']}")
        lines.append(f"loso_coverage_rate_95ppi: {payload['loso_coverage_rate_95ppi']}")
    lines.append("")
    lines.append("B10_at_35_by_model:")
    for row in payload.get("b10_at_35_by_model", []):
        lines.append(
            f"  {row['model']}: median={row['b10_q50']:.4f}, "
            f"95%=[{row['b10_q025']:.4f}, {row['b10_q975']:.4f}]"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(
    out_file: Path,
    source_kind: str,
    csv_path: Path,
    strict_validation_passed: Optional[bool],
    waic_dic: Dict[str, float],
    accept_rates: np.ndarray,
    rhat: np.ndarray,
    ess: np.ndarray,
    delta_t_use: float,
    picm_c_uncertainty_method: str,
    lognormal_uncertainty_method: str,
) -> None:
    """Write run metadata and key metrics for reproducibility."""
    low_delta_note = (
        f"NOTE: DeltaT estimates below {EXTRAPOLATION_CUTOFF:.0f}C are extrapolations beyond observed failures "
        f"(observed range {OBSERVED_DOMAIN_MIN:.0f}-{OBSERVED_DOMAIN_MAX:.0f}C)."
    )
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"source_kind: {source_kind}",
        f"input_file: {str(csv_path.resolve())}",
        f"strict_real_validation_passed: {strict_validation_passed}",
        f"delta_T_use: {delta_t_use}",
        f"target_deltaT_values: {TARGET_DELTA_TS.tolist()}",
        f"mission_cycle_targets: {MISSION_CYCLE_TARGETS}",
        f"seeds: SEED_OPT={SEED_OPT}, SEED_MCMC={SEED_MCMC}, SEED_BOOT={SEED_BOOT}, SEED_PRED={SEED_PRED}",
        "k_prior_effective_form: Gamma(shape,rate) truncated to k > 1.0",
        f"sensitivity_uncertainty_picm_c: {picm_c_uncertainty_method}",
        f"sensitivity_uncertainty_lognormal: {lognormal_uncertainty_method}",
        f"WAIC: {waic_dic['waic']}",
        f"p_WAIC: {waic_dic['p_waic']}",
        f"DIC_supplementary: {waic_dic['dic']}",
        f"p_D: {waic_dic['p_dic']}",
        "criteria_comparability_note: WAIC/DIC reported for PICM-L only; AIC/BIC used for cross-model comparison.",
        f"accept_rates: {', '.join([f'{a:.6f}' for a in accept_rates])}",
        f"Rhat: k={rhat[0]:.6f}, beta1={rhat[1]:.6f}, beta0={rhat[2]:.6f}",
        f"ESS: k={ess[0]:.2f}, beta1={ess[1]:.2f}, beta0={ess[2]:.2f}",
        low_delta_note,
    ]
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(
    data: ReliabilityData,
    mle: Dict[str, object],
    firth: Dict[str, object],
    picm_c: Dict[str, object],
    lognorm: Dict[str, object],
    post_summary: pd.DataFrame,
    rhat: np.ndarray,
    ess: np.ndarray,
    accept_rates: np.ndarray,
    waic_dic: Dict[str, float],
    table_d: pd.DataFrame,
    lognormal_b10_use: float,
) -> None:
    """Print concise interpretive study summary."""
    print("\n================= Optocoupler Reliability Study Summary =================")
    print(f"Dataset: {data.n} units, stresses={[int(v) for v in data.stress_levels]}, cycle time={CYCLE_TIME_HOURS:g} h")
    print(f"Failures={int(np.sum(data.event))}, Censored={int(np.sum(1 - data.event))}")

    print("\nPICM-L (Weibull) parameter estimates:")
    print(f"  MLE   : k={mle['theta'][0]:.4f}, beta1={mle['theta'][1]:.4f}, beta0={mle['theta'][2]:.4f}")
    print(
        f"  Firth (consistency check): k={firth['theta'][0]:.4f}, beta1={firth['theta'][1]:.4f}, "
        f"beta0={firth['theta'][2]:.4f} - in agreement with MLE"
    )
    print(
        "    Firth diagnostic counters: "
        f"ridge_trigger_count={int(firth.get('ridge_trigger_count', len(firth.get('ridge_events', []))))}, "
        f"rejected_hessian_count={int(firth.get('rejected_hessian_count', firth.get('non_pd_count', 0)))}"
    )
    print("  Bayes :")
    for _, r in post_summary.iterrows():
        print(
            f"    {r['parameter']}: mean={r['mean']:.4f}, median={r['median']:.4f}, "
            f"95% ETI=({r['eti_2.5%']:.4f}, {r['eti_97.5%']:.4f})"
        )

    print("\nMCMC diagnostics:")
    print(f"  Chain acceptance rates: {', '.join([f'{a:.3f}' for a in accept_rates])}")
    print(f"  R-hat: k={rhat[0]:.4f}, beta1={rhat[1]:.4f}, beta0={rhat[2]:.4f}")
    print(f"  ESS:   k={ess[0]:.1f}, beta1={ess[1]:.1f}, beta0={ess[2]:.1f}")
    if np.any((accept_rates < MCMC_ACCEPT_LOWER) | (accept_rates > MCMC_ACCEPT_UPPER)):
        print(f"  WARNING: At least one chain acceptance rate is outside [{MCMC_ACCEPT_LOWER:.2f}, {MCMC_ACCEPT_UPPER:.2f}].")
    if np.any(rhat > 1.01):
        print("  WARNING: At least one R-hat exceeds 1.01.")
    if np.any(ess < 1000.0):
        print("  WARNING: At least one retained ESS is < 1000.")

    print("\nBayesian model criteria (PICM-L):")
    print(f"  WAIC (primary) = {waic_dic['waic']:.3f}, p_WAIC = {waic_dic['p_waic']:.3f}")
    print(f"  DIC  (supp.)   = {waic_dic['dic']:.3f}, p_D = {waic_dic['p_dic']:.3f}")
    print("  Note: WAIC/DIC are PICM-L-only and are not used as cross-model criteria.")

    print("\nSensitivity models:")
    print(
        f"  PICM-C: k={picm_c['theta'][0]:.4f}, beta1={picm_c['theta'][1]:.4f}, "
        f"beta0={picm_c['theta'][2]:.4f}, beta2={picm_c['theta'][3]:.4f}; "
        f"AIC={picm_c['aic']:.2f}, BIC={picm_c['bic']:.2f}"
    )
    print(
        f"  Lognormal AFT: sigma={lognorm['theta'][0]:.4f}, beta1={lognorm['theta'][1]:.4f}, beta0={lognorm['theta'][2]:.4f}; "
        f"AIC={lognorm['aic']:.2f}, BIC={lognorm['bic']:.2f}; reference-case B10={lognormal_b10_use:.2f} cycles"
    )

    print("\nUse-condition B10 (cycles):")
    for _, r in table_d.iterrows():
        lo = float(r["lower_95"]) if pd.notna(r["lower_95"]) else np.nan
        hi = float(r["upper_95"]) if pd.notna(r["upper_95"]) else np.nan
        note = str(r.get("interval_note", ""))
        if np.isfinite(lo) and np.isfinite(hi):
            print(
                f"  {r['method']}: estimate={r['B10_cycles_estimate']:.3f}, "
                f"95% interval=({lo:.3f}, {hi:.3f})"
            )
        else:
            print(f"  {r['method']}: estimate={r['B10_cycles_estimate']:.3f} ({note})")
    print("==========================================================================\n")


# =============================================================================
# Main execution
# =============================================================================


def _cycle_tick_plain(x: float, _pos: Optional[int] = None) -> str:
    """Readable integer tick labels for cycle counts."""
    if abs(float(x)) >= 1000:
        return f"{float(x):,.0f}"
    return f"{float(x):g}"


def _cycle_tick_compact(x: float, _pos: Optional[int] = None) -> str:
    """Compact cycle-count ticks for small multi-panel figures."""
    if abs(float(x)) >= 1000:
        return f"{float(x) / 1000:g}k"
    return f"{float(x):g}"


def _prior_quantiles_for_parameter(param: str, cfg: Dict[str, float]) -> Tuple[float, float, float]:
    """Return baseline prior 2.5%, 50%, and 97.5% quantiles for a PICM-L parameter."""
    if param == "k":
        dist = stats.gamma(a=cfg["k_gamma_shape"], scale=1.0 / cfg["k_gamma_rate"])
        lower = cfg["k_trunc_lower"]
        cdf_lower = dist.cdf(lower)
        return tuple(float(dist.ppf(cdf_lower + p * (1.0 - cdf_lower))) for p in (0.025, 0.5, 0.975))
    if param == "beta1":
        a = (cfg["beta1_trunc_lower"] - cfg["beta1_mean"]) / cfg["beta1_sd"]
        dist = stats.truncnorm(a=a, b=np.inf, loc=cfg["beta1_mean"], scale=cfg["beta1_sd"])
        return tuple(float(dist.ppf(p)) for p in (0.025, 0.5, 0.975))
    if param == "beta0":
        dist = stats.norm(loc=cfg["beta0_mean"], scale=cfg["beta0_sd"])
        return tuple(float(dist.ppf(p)) for p in (0.025, 0.5, 0.975))
    raise ValueError(f"Unknown parameter: {param}")


def plot_main_reference_case_b10_comparison(
    use_b10_samples: np.ndarray,
    b10_mle: Tuple[float, float, float],
    b10_firth: Tuple[float, float, float],
    b10_prof_unc: Tuple[float, float, float],
    b10_prof_bc: Tuple[float, float, float],
    delta_t_use: float,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Publication-readable Fig. 4: forest plot with numeric interval summary."""
    def fmt_cycles(value: float) -> str:
        return f"{value:,.0f}" if np.isfinite(value) else "-"

    bayes = (
        float(np.median(use_b10_samples)),
        float(np.quantile(use_b10_samples, 0.025)),
        float(np.quantile(use_b10_samples, 0.975)),
    )
    rows = [
        {
            "method": "Bayesian posterior",
            "role": "Primary ETI",
            "values": bayes,
            "color": PLOT_COLORS["bayes_fit"],
        },
        {
            "method": "Profile likelihood",
            "role": "Bartlett CI",
            "values": b10_prof_bc,
            "color": "#d97706",
        },
        {
            "method": "MLE delta method",
            "role": "Asymptotic",
            "values": b10_mle,
            "color": PLOT_COLORS["mle"],
        },
        {
            "method": "Profile likelihood",
            "role": "Uncorr. CI",
            "values": b10_prof_unc,
            "color": "#8b5cf6",
        },
        {
            "method": "Firth estimate",
            "role": "Point check",
            "values": (float(b10_firth[0]), np.nan, np.nan),
            "color": "#8f4b3e",
        },
    ]
    fig = plt.figure(figsize=(IEEE_TWO_COL, 4.72))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.42], hspace=0.42)
    ax = fig.add_subplot(grid[0, 0])
    ax_tbl = fig.add_subplot(grid[1, 0])
    yvals = np.arange(len(rows))[::-1]
    finite_limits: List[float] = []
    for row in rows:
        vals = row["values"]
        finite_limits.extend(float(v) for v in vals if np.isfinite(v))
    xmax = max(9000.0, math.ceil(max(finite_limits) * 1.06 / 1000.0) * 1000.0)

    for idx, y in enumerate(yvals):
        band_color = "#f8fafc" if idx % 2 == 0 else "#ffffff"
        ax.axhspan(y - 0.42, y + 0.42, color=band_color, zorder=0)

    bayes_median = float(bayes[0])
    ax.axvline(
        bayes_median,
        color="#9ca3af",
        lw=0.9,
        ls="--",
        alpha=0.85,
        zorder=1,
    )
    ax.text(
        bayes_median,
        len(rows) - 0.35,
        "Bayes med.",
        fontsize=6.0,
        color="#6b7280",
        rotation=90,
        ha="right",
        va="top",
    )

    for y, row in zip(yvals, rows):
        vals = row["values"]
        color = str(row["color"])
        est = float(vals[0])
        lo = float(vals[1]) if len(vals) > 1 and np.isfinite(vals[1]) else np.nan
        hi = float(vals[2]) if len(vals) > 2 and np.isfinite(vals[2]) else np.nan
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.errorbar(
                est,
                y,
                xerr=np.array([[est - lo], [hi - est]], dtype=float),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.15,
                capsize=4.2,
                capthick=1.35,
                markersize=5.2,
                zorder=3,
            )
        else:
            ax.plot(est, y, "o", color=color, ms=5.2, zorder=3)

    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, 2000.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_cycle_tick_plain))
    ax.set_yticks(yvals)
    ax.set_yticklabels([str(r["method"]) for r in rows])
    ax.set_xlabel(f"B10 at Delta T = {delta_t_use:.0f} deg C (cycles)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(True, axis="x", color="#d8dee6", alpha=0.58, linewidth=0.75)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", colors="#374151", labelsize=7.6, length=3.2, width=0.8)
    ax.tick_params(axis="y", colors="#1f2937", labelsize=7.4, length=3.2, width=0.8)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("semibold")
    ax.set_title("Reference-case B10 interval comparison", loc="left", fontsize=8.4, pad=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)

    ax_tbl.axis("off")
    col_method = 0.02
    col_role = 0.33
    col_est = 0.58
    col_interval = 0.72
    col_width = 0.98
    headers = [
        ("Method", col_method, "left"),
        ("Role", col_role, "left"),
        ("Estimate", col_est, "right"),
        ("95% interval", col_interval, "left"),
        ("Width", col_width, "right"),
    ]
    ax_tbl.hlines([0.88, 0.18], 0.0, 1.0, color="#d1d5db", lw=0.8)
    for label, x, ha in headers:
        ax_tbl.text(x, 0.93, label, fontsize=6.6, color="#4b5563", weight="bold", ha=ha, va="center")

    table_y = np.linspace(0.76, 0.28, len(rows))
    for idx, (row, y) in enumerate(zip(rows, table_y)):
        if idx % 2 == 0:
            ax_tbl.axhspan(y - 0.055, y + 0.055, color="#f8fafc", zorder=0)
        vals = row["values"]
        color = str(row["color"])
        est = float(vals[0])
        lo = float(vals[1]) if len(vals) > 1 and np.isfinite(vals[1]) else np.nan
        hi = float(vals[2]) if len(vals) > 2 and np.isfinite(vals[2]) else np.nan
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            interval_txt = f"{fmt_cycles(lo)}-{fmt_cycles(hi)}"
            width_txt = fmt_cycles(hi - lo)
        else:
            interval_txt = "point only"
            width_txt = "-"
        ax_tbl.scatter([col_method], [y], s=16, color=color, zorder=3, clip_on=False)
        ax_tbl.text(col_method + 0.025, y, str(row["method"]), fontsize=6.1, color="#111827", ha="left", va="center")
        ax_tbl.text(col_role, y, str(row["role"]), fontsize=6.1, color="#374151", ha="left", va="center")
        ax_tbl.text(col_est, y, fmt_cycles(est), fontsize=6.1, color="#111827", ha="right", va="center")
        ax_tbl.text(col_interval, y, interval_txt, fontsize=6.1, color="#111827", ha="left", va="center")
        ax_tbl.text(col_width, y, width_txt, fontsize=6.1, color="#111827", ha="right", va="center")
    ax_tbl.text(
        0.0,
        0.05,
        (
            "Horizontal bars show 95% intervals where used for inference; "
            "Firth is displayed only as a consistency-check point estimate."
        ),
        fontsize=6.1,
        color="#4b5563",
        ha="left",
        va="center",
    )
    ax_tbl.set_xlim(0, 1.0)
    ax_tbl.set_ylim(0, 1.0)

    fig.subplots_adjust(left=0.24, right=0.985, top=0.94, bottom=0.08)
    save_figure_dual(
        fig,
        out_dir,
        "main_reference_case_b10_comparison",
        manifest,
        "Main figure",
        "Reference-case B10 estimates and interval summaries.",
    )
    plt.close(fig)


def plot_prior_sensitivity_b10_35c(
    b10_35_by_prior: Dict[str, np.ndarray],
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Publication-readable Fig. S5: prior sensitivity intervals at 20, 25, and 35 deg C."""
    summary_path = TAB_DIR / "table_prior_sensitivity_b10_extended.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        rows: List[Dict[str, object]] = []
        for key, arr in b10_35_by_prior.items():
            draws = np.asarray(arr, dtype=float)
            rows.append(
                {
                    "prior_setting": key,
                    "delta_T_C": float(DELTA_T_REFERENCE),
                    "B10_q2_5": float(np.quantile(draws, 0.025)),
                    "B10_q50": float(np.quantile(draws, 0.5)),
                    "B10_q97_5": float(np.quantile(draws, 0.975)),
                }
            )
        df = pd.DataFrame(rows)

    deltas = [20.0, 25.0, 35.0]
    available = sorted(float(v) for v in df["delta_T_C"].dropna().unique())
    deltas = [d for d in deltas if any(np.isclose(d, available))]
    settings = [s for s in PRIOR_SENSITIVITY_KEYS if s in set(df["prior_setting"].astype(str))]
    colors = {"baseline": PLOT_COLORS["bayes_fit"], "diffuse": "#b7352d", "conservative": "#2a7f73"}
    labels = {"baseline": "Baseline", "diffuse": "Diffuse", "conservative": "Conservative"}

    fig, axes = plt.subplots(1, len(deltas), figsize=(7.8, 2.75), sharey=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)
    for ax, delta in zip(axes_arr, deltas):
        sub = df[np.isclose(df["delta_T_C"].astype(float), delta)].copy()
        sub = sub.set_index("prior_setting").loc[settings]
        yvals = np.arange(len(settings))[::-1]
        for y, setting in zip(yvals, settings):
            row = sub.loc[setting]
            lo = float(row["B10_q2_5"])
            med = float(row["B10_q50"])
            hi = float(row["B10_q97_5"])
            color = colors.get(setting, PLOT_COLORS["aux"])
            ax.hlines(y, lo, hi, color=color, lw=2.2)
            ax.plot([lo, hi], [y, y], "|", color=color, ms=7.5, mew=1.2)
            ax.plot(med, y, "o", color=color, ms=4.6)
            # Exact medians are tabulated; omitting in-panel labels avoids title crowding.
        ax.set_xscale("log")
        ax.set_xlim(1300, 60000)
        ax.set_xticks([2000, 10000, 50000])
        ax.xaxis.set_major_formatter(FuncFormatter(_cycle_tick_compact))
        ax.set_title(f"{delta:.0f} deg C", pad=5)
        ax.set_xlabel("B10 cycles")
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes_arr[0].set_yticks(np.arange(len(settings))[::-1])
    axes_arr[0].set_yticklabels([labels.get(s, s) for s in settings])
    axes_arr[0].set_ylabel("Prior setting")
    fig.tight_layout()
    save_figure_dual(
        fig,
        out_dir,
        "supp_prior_sensitivity_b10_35C",
        manifest,
        "Supplementary",
        "Prior-sensitivity B10 intervals at 20, 25, and 35 deg C.",
    )
    plt.close(fig)


def plot_supp_leave_one_stress_out(
    data: ReliabilityData,
    out_dir: Path,
    manifest: List[Dict[str, str]],
    enable: bool = True,
    loso_table: Optional[pd.DataFrame] = None,
) -> None:
    """Publication-readable Fig. S4: LOSO adequacy without clipped labels."""
    if not enable:
        warnings.warn("LOSO figure disabled by run control.")
        return
    if loso_table is None or loso_table.empty:
        warnings.warn("LOSO table is empty; skipping LOSO figure.")
        return
    chk = loso_table.sort_values("held_out_stress").reset_index(drop=True).copy()
    x = chk["observed_failure_fraction"].astype(float).to_numpy()
    y = chk["predicted_failure_fraction"].astype(float).to_numpy()
    lo = chk["predicted_pi_2_5"].astype(float).to_numpy()
    hi = chk["predicted_pi_97_5"].astype(float).to_numpy()
    stress = chk["held_out_stress"].astype(float).to_numpy()
    failed = chk["observed_failed"].astype(int).to_numpy()
    total = chk["observed_total"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(4.95, 3.95))
    ax.plot([0, 1], [0, 1], ls="--", color=PLOT_COLORS["aux"], lw=1.2, label="45-degree reference")
    yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=PLOT_COLORS["bayes_fit"],
        ecolor=PLOT_COLORS["bayes_fit"],
        elinewidth=1.45,
        capsize=3.2,
        ms=5.8,
        label="LOSO prediction (95% PI)",
        zorder=3,
    )
    offsets = {
        75.0: (0.025, -0.035),
        100.0: (-0.035, -0.055),
        125.0: (-0.035, -0.145),
        150.0: (-0.035, -0.105),
    }
    for xi, yi, st, f, n in zip(x, y, stress, failed, total):
        if st in (25.0, 50.0):
            continue
        dx, dy = offsets.get(float(st), (0.02, 0.02))
        ax.text(
            xi + dx,
            yi + dy,
            f"{st:.0f} deg C ({f}/{n})",
            fontsize=7.6,
            color="#263241",
            weight="semibold",
            ha="left" if dx >= 0 else "right",
            va="center",
        )
    ax.text(0.03, 0.055, "25 and 50 deg C (0/8)", fontsize=7.6, color="#263241", weight="semibold", ha="left", va="center")
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Observed failure fraction")
    ax.set_ylabel("Predicted failure fraction")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure_dual(
        fig,
        out_dir,
        "supp_leave_one_stress_out",
        manifest,
        "Supplementary",
        "Leave-one-stress-out observed versus predicted failure fractions.",
    )
    plt.close(fig)


def plot_supp_prior_vs_posterior_refined(
    bayes_samples: np.ndarray,
    out_dir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Publication-readable Fig. S6: prior-vs-posterior intervals, not crowded densities."""
    samples = np.asarray(bayes_samples, dtype=float)
    cfg = PRIOR_SETTINGS[BASELINE_PRIOR_KEY]
    params = [("k", "Weibull shape k", 0), ("beta1", "Stress slope beta1", 1), ("beta0", "Intercept beta0", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    for ax, (param, title, col) in zip(np.asarray(axes, dtype=object).reshape(-1), params):
        prior_lo, prior_med, prior_hi = _prior_quantiles_for_parameter(param, cfg)
        post = samples[:, col]
        post_lo, post_med, post_hi = (float(np.quantile(post, q)) for q in (0.025, 0.5, 0.975))
        rows = [
            ("Prior", prior_lo, prior_med, prior_hi, "#8c4a3a"),
            ("Posterior", post_lo, post_med, post_hi, PLOT_COLORS["bayes_fit"]),
        ]
        yvals = [1, 0]
        for y, (_label, lo, med, hi, color) in zip(yvals, rows):
            ax.hlines(y, lo, hi, color=color, lw=2.6)
            ax.plot([lo, hi], [y, y], "|", color=color, ms=8.0, mew=1.3)
            ax.plot(med, y, "o", color=color, ms=5.0)
            ax.text(med, y + 0.15, f"{med:.2f}", color=color, fontsize=7.2, ha="center")
        ax.set_yticks(yvals)
        ax.set_yticklabels(["Prior", "Posterior"])
        ax.set_ylim(-0.55, 1.55)
        ax.set_title(title, pad=5)
        ax.set_xlabel("Parameter value")
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure_dual(
        fig,
        out_dir,
        "supp_prior_vs_posterior_refined",
        manifest,
        "Supplementary",
        "Baseline prior versus data-informed posterior interval summaries.",
    )
    plt.close(fig)


def write_figure_qa_report(manifest_df: pd.DataFrame) -> Path:
    """Write a reproducibility-oriented figure QA report from the final manifest."""
    qa_dir = OUT_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    zero_size: List[str] = []
    for _, row in manifest_df.iterrows():
        cols = ["png_relpath"]
        if str(row.get("paper_png_relpath", "") or ""):
            cols.append("paper_png_relpath")
        for col in cols:
            path = OUT_DIR / str(row[col])
            if not path.exists():
                missing.append(str(path.relative_to(OUT_DIR)))
            elif path.stat().st_size <= 0:
                zero_size.append(str(path.relative_to(OUT_DIR)))
    lines = [
        "# Figure QA Report for V5 Outputs",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Manifest rows: {len(manifest_df)}",
        f"Missing files: {len(missing)}",
        f"Zero-size files: {len(zero_size)}",
        "",
        "## Publication-readability changes built into V5",
        "",
        "- Fig. 4 uses a cleaner interval-summary forest plot with improved label spacing.",
        "- Fig. S4 uses unclipped leave-one-stress-out labels.",
        "- Fig. S5 uses prior-sensitivity B10 intervals at 20, 25, and 35 deg C.",
        "- Fig. S6 uses prior-versus-posterior interval summaries instead of crowded density panels.",
        "- Fig. S11 reports high-side 175 and 200 deg C model-conditional extrapolation intervals.",
        "",
        "## Checks",
        "",
        "- All manifest-listed PNG files are checked for existence and nonzero size.",
        "- `figures/contact_sheet.png` is generated from the same manifest for manual visual inspection.",
        "- `figures/all/` keeps source figure PNG names.",
        "- `figures/paper/` is synchronized from the manifest slots.",
    ]
    if missing:
        lines.extend(["", "## Missing files", ""])
        lines.extend(f"- {item}" for item in missing)
    if zero_size:
        lines.extend(["", "## Zero-size files", ""])
        lines.extend(f"- {item}" for item in zero_size)
    report_path = qa_dir / "figure_qa_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if it exists."""
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree_contents(src: Path, dst: Path) -> None:
    """Copy directory contents into an existing package directory."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def build_repeatability_package() -> Path:
    """Create a fresh-submission GitHub package with one executable V5 script."""
    script_path = Path(__file__).resolve()
    source_root = script_path.parent.parent
    package_root = (
        source_root.parent
        / f"github_standard_submission_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    subdirs = [
        "code",
        "data",
        "tables",
        "figures",
        "qa",
    ]
    for subdir in subdirs:
        (package_root / subdir).mkdir(parents=True, exist_ok=True)

    _copy_if_exists(script_path, package_root / "code" / script_path.name)
    _copy_if_exists(resolve_input_csv(INPUT_CSV), package_root / "data" / INPUT_CSV.name)
    _copy_if_exists(
        OUT_DIR / "optocoupler_ttf_unit_level_canonical.csv",
        package_root / "data" / "optocoupler_ttf_unit_level_canonical.csv",
    )
    _copy_tree_contents(TAB_DIR, package_root / "tables")
    _copy_tree_contents(FIG_DIR, package_root / "figures")
    _copy_tree_contents(OUT_DIR / "qa", package_root / "qa")
    (package_root / "requirements.txt").write_text(
        "\n".join(["numpy", "pandas", "scipy", "matplotlib"]) + "\n",
        encoding="utf-8",
    )
    (package_root / "README.md").write_text(
        "\n".join(
            [
                "# Optocoupler Thermal-Cycling Reliability V5",
                "",
                (
                    "This repository contains the data, analysis code, tables, and figures "
                    "for the V5 optocoupler thermal-cycling reliability analysis."
                ),
                "",
                "The only executable analysis script is:",
                "",
                "`code/optocoupler_thermal_reliability_study_pubrev_v5.py`",
                "",
                "Run it from the package root with:",
                "",
                "```bash",
                "python code/optocoupler_thermal_reliability_study_pubrev_v5.py",
                "```",
                "",
                (
                    "The script reads `data/optocoupler_ttf_unit_level.csv`, performs the "
                    "censored reliability analysis, generates all tables and figures, "
                    "writes the figure manifest/contact sheet/QA report, and creates a "
                    "clean GitHub package."
                ),
                "",
                "Included outputs:",
                "",
                "- `data/`: raw and canonical unit-level data.",
                "- `tables/`: generated result tables.",
                "- `figures/all/`: source figure PNGs using the analysis figure names.",
                "- `figures/paper/`: paper-ready figure PNGs using the figure-slot names.",
                "- `figures/figure_manifest.csv` and `figures/contact_sheet.png`: figure index and visual QA sheet.",
                "- `qa/`: figure QA report.",
                "",
                (
                    "Interpretation note: 15 deg C and 200 deg C results are "
                    "model-conditional extrapolations, not independently validated "
                    "test conditions."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_rows = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(package_root)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(package_root / "PACKAGE_MANIFEST.csv", index=False)
    return package_root


def main() -> None:
    if DELTA_T_USE <= 0:
        raise ValueError("DELTA_T_USE must be > 0.")
    if CYCLE_TIME_HOURS <= 0:
        raise ValueError("CYCLE_TIME_HOURS must be > 0.")
    if np.any(TARGET_DELTA_TS <= 0):
        raise ValueError("TARGET_DELTA_TS must contain only positive values.")
    if np.any(np.asarray(MISSION_CYCLE_TARGETS, dtype=float) <= 0):
        raise ValueError("MISSION_CYCLE_TARGETS must contain only positive values.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ALL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    reset_figure_outputs()
    reset_table_outputs()

    setup_plot_style()
    warnings.simplefilter("always", UserWarning)

    # 1) Load and validate dataset
    df, source_kind, csv_path = load_input_dataset(
        input_csv=INPUT_CSV,
        cycle_time_hours=CYCLE_TIME_HOURS,
    )
    validation_info = validate_dataset(
        df=df,
        cycle_time_hours=CYCLE_TIME_HOURS,
        enforce_real_failure_pattern=ENFORCE_REAL_FAILURE_PATTERN,
        pattern_mismatch_is_error=REAL_PATTERN_MISMATCH_IS_ERROR,
    )
    print_data_provenance(source_kind=source_kind, csv_path=csv_path, df=df, validation_info=validation_info)

    data = prepare_data(df)
    df.to_csv(TAB_DIR / "dataset_unit_level.csv", index=False)
    write_canonical_unit_level_csv(data, OUT_DIR / "optocoupler_ttf_unit_level_canonical.csv")

    baseline_prior_cfg = get_prior_config(BASELINE_PRIOR_KEY)
    prior_hyper_table = build_prior_hyperparameter_table()
    prior_effective_table = build_effective_k_prior_summary_table()
    prior_hyper_table.to_csv(TAB_DIR / "prior_hyperparameters.csv", index=False)
    prior_effective_table.to_csv(TAB_DIR / "prior_effective_summary.csv", index=False)
    (TAB_DIR / "prior_hyperparameters.json").write_text(
        json.dumps(prior_hyper_table.to_dict(orient="records"), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    table_a = make_table_a(data)
    table_a.to_csv(TAB_DIR / "table_A_data_summary.csv", index=False)
    censoring_summary = build_censoring_summary_by_stress(data)
    censoring_summary.to_csv(TAB_DIR / "censoring_summary_by_stress.csv", index=False)

    # 2) PICM-L MLE and Firth
    mle = fit_picm_l_mle(data, n_starts=N_OPT_STARTS_MAIN, seed=SEED_OPT)
    firth = fit_picm_l_firth(data, mle_theta=mle["theta"], n_starts=max(20, N_OPT_STARTS_MAIN // 2), seed=SEED_OPT + 1)

    ridge_used_count = len(firth["ridge_events"])
    if ridge_used_count > 0:
        print(
            f"WARNING: Firth observed-information ridge stabilization triggered {ridge_used_count} times "
            f"(max ridge={max(firth['ridge_events']):.3e})."
        )
    if firth["non_pd_count"] > 0:
        print(f"WARNING: Firth penalized objective rejected {firth['non_pd_count']} non-PD Hessian evaluations.")

    # 3) PICM-C sensitivity
    picm_c = fit_picm_c_sensitivity(
        data,
        n_starts=max(20, N_OPT_STARTS_MAIN // 2),
        seed=SEED_OPT + 2,
        prior_cfg=baseline_prior_cfg,
    )

    # 4) Lognormal AFT sensitivity
    lognorm = fit_lognormal_aft(data, n_starts=max(20, N_OPT_STARTS_MAIN // 2), seed=SEED_OPT + 3)

    # 5) Bayesian PICM-L adaptive MH
    starts = make_picm_l_mcmc_starts(mle["theta"], prior_cfg=baseline_prior_cfg)
    for i in range(3):
        if not np.isfinite(logposterior_picm_l(starts[i], data, prior_cfg=baseline_prior_cfg)):
            starts[i] = np.array(
                [
                    baseline_prior_cfg["k_trunc_lower"] + 1.0 + 0.1 * i,
                    max(baseline_prior_cfg["beta1_mean"], baseline_prior_cfg["beta1_trunc_lower"] + 0.2 + 0.1 * i),
                    baseline_prior_cfg["beta0_mean"] - 0.6 * i,
                ],
                dtype=float,
            )

    mh = run_adaptive_mh_picm_l(
        data=data,
        starts=starts,
        seeds=SEED_MCMC,
        burnin=MCMC_BURNIN,
        keep=MCMC_KEEP,
        adapt_block=MCMC_ADAPT_BLOCK,
        prior_cfg=baseline_prior_cfg,
    )
    chains = mh["chains"]  # shape (m, keep, 3)
    accept_rates = mh["accept_rates"]
    samples = chains.reshape(-1, chains.shape[-1])

    # 6) Posterior summaries + diagnostics
    post_summary = summarize_posterior(samples, names=["k", "beta1", "beta0"])
    rhat = compute_rhat(chains)
    ess = np.array([effective_sample_size(chains[:, :, j]) for j in range(chains.shape[2])], dtype=float)

    if np.any((accept_rates < MCMC_ACCEPT_LOWER) | (accept_rates > MCMC_ACCEPT_UPPER)):
        warnings.warn(
            f"At least one post-adaptation sampling acceptance rate is outside "
            f"[{MCMC_ACCEPT_LOWER:.2f}, {MCMC_ACCEPT_UPPER:.2f}]."
        )
    if np.any(rhat > 1.01):
        warnings.warn("At least one R-hat exceeds 1.01.")
    if np.any(ess < 1000.0):
        warnings.warn("At least one retained ESS is < 1000; consider longer chains.")

    param_checks = run_parameterization_consistency_checks(data=data, mle_theta=mle["theta"], bayes_samples=samples)
    param_checks.to_csv(TAB_DIR / "parameterization_consistency_checks.csv", index=False)
    if not bool(np.all(param_checks["passed"].to_numpy(dtype=bool))):
        failed = param_checks.loc[~param_checks["passed"].astype(bool), "check_name"].tolist()
        raise RuntimeError(f"Parameterization consistency checks failed: {failed}")

    # 7) WAIC and DIC (DIC supplementary)
    waic_dic = compute_waic_dic(samples, data)

    # 8) Uncertainty draws for sensitivity models (Laplace approximation)
    picm_c_unc = draw_picm_c_laplace(picm_c, n_draws=N_SENSITIVITY_DRAWS, seed=SEED_OPT + 41)
    lognorm_unc = draw_lognormal_laplace(lognorm, n_draws=N_SENSITIVITY_DRAWS, seed=SEED_OPT + 42)
    picm_c_draws = picm_c_unc["draws"]
    lognorm_draws = lognorm_unc["draws"]

    # 9) Profile likelihood and Bartlett correction for the reference-case B10 (Delta T = 35 deg C)
    profile = profile_likelihood_b10(
        data=data,
        mle_theta=mle["theta"],
        mle_loglik=mle["loglik"],
        delta_t_use=DELTA_T_USE,
        n_grid=N_PROFILE_GRID,
        seed=SEED_BOOT + 101,
    )
    logpsi_hat = logpsi_from_theta(mle["theta"], DELTA_T_USE)
    bart = bartlett_factor_bootstrap(
        data=data,
        mle_theta=mle["theta"],
        logpsi_hat=logpsi_hat,
        delta_t_use=DELTA_T_USE,
        n_boot=N_BARTLETT_BOOT,
        seed=SEED_BOOT,
    )
    bartlett_factor = bart["bartlett_factor"]
    lr_corr = profile["lr_grid"] / bartlett_factor
    left_bc, right_bc = find_profile_ci(profile["logpsi_grid"], lr_corr, profile["q95"])
    ci_bc = (float(np.exp(left_bc)), float(np.exp(right_bc)))

    # 10) Posterior predictive reliability quantities
    post_pred = posterior_predictive_quantities(samples, data.stress_levels, delta_t_use=DELTA_T_USE)
    failure_fraction_check = compute_failure_fraction_crosscheck(data=data, bayes_samples=samples, seed=SEED_PRED + 77)
    failure_fraction_check.to_csv(TAB_DIR / "failure_fraction_crosscheck.csv", index=False)

    print("\nPICM-L failure-fraction cross-check by stress:")
    for _, r in failure_fraction_check.iterrows():
        print(
            f"  DeltaT={int(r['delta_T'])}C: observed={r['observed_failure_fraction']:.3f}, "
            f"pred_median={r['predicted_median_failure_fraction']:.3f}, "
            f"95% PI=({r['predicted_pi_2.5%']:.3f}, {r['predicted_pi_97.5%']:.3f}), "
            f"inside_PI={bool(r['observed_within_95ppi'])}"
        )

    loso_table = compute_loso_predictive_adequacy_picm_l(data=data, prior_cfg=baseline_prior_cfg, max_stress=LOSO_MAX_STRESS)
    loso_table.to_csv(TAB_DIR / "loso_predictive_adequacy_picm_l.csv", index=False)
    write_loso_summary_outputs(loso_table, TAB_DIR)

    # 11) Low-deltaT extrapolation predictions and mission metrics
    if len(MISSION_CYCLE_TARGETS) < 2:
        raise ValueError("MISSION_CYCLE_TARGETS must contain at least two cycle targets.")
    mission_1 = float(MISSION_CYCLE_TARGETS[0])
    mission_2 = float(MISSION_CYCLE_TARGETS[1])

    prior_sens = run_prior_sensitivity_picm_l(
        data=data,
        prior_keys=PRIOR_SENSITIVITY_KEYS,
        mission_cycles=[mission_1, mission_2],
    )
    prior_sens_summary = prior_sens["summary_table"].copy()
    prior_sens_summary.to_csv(TAB_DIR / "prior_sensitivity_summary.csv", index=False)
    (TAB_DIR / "prior_sensitivity_summary.json").write_text(
        json.dumps(prior_sens_summary.to_dict(orient="records"), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    prior_sens_b10_extended = build_prior_sensitivity_b10_extended(
        draws_by_prior=prior_sens["draws_by_prior"],
        delta_targets=PRIOR_SENS_DELTA_TARGETS,
    )
    prior_sens_b10_extended.to_csv(TAB_DIR / "table_prior_sensitivity_b10_extended.csv", index=False)

    prior_only_draws = sample_picm_l_prior(
        n_draws=N_PRIOR_ONLY_DRAWS,
        prior_key=BASELINE_PRIOR_KEY,
        seed=SEED_PRED + 505,
    )
    table_prior_vs_data, prior_b10_only, post_b10_for_prior_compare = build_prior_vs_data_b10(
        posterior_samples=samples,
        prior_draws=prior_only_draws,
        delta_targets=PRIOR_VS_DATA_TARGETS,
    )
    table_prior_vs_data.to_csv(TAB_DIR / "table_prior_vs_data_b10.csv", index=False)
    plot_prior_vs_data_b10(table_prior_vs_data, SUPP_FIG_DIR)

    pred_picm_l_targets = predict_picm_l_quantities(samples, TARGET_DELTA_TS)
    pred_picm_c_targets = predict_picm_c_quantities(picm_c_draws, TARGET_DELTA_TS, xbar=data.xbar)
    pred_lognorm_targets = predict_lognormal_quantities(lognorm_draws, TARGET_DELTA_TS)

    pred_picm_l_grid = predict_picm_l_quantities(samples, B10_GRID_DELTA_T)
    pred_picm_c_grid = predict_picm_c_quantities(picm_c_draws, B10_GRID_DELTA_T, xbar=data.xbar)
    pred_lognorm_grid = predict_lognormal_quantities(lognorm_draws, B10_GRID_DELTA_T)
    pred_picm_l_high_grid = predict_picm_l_quantities(samples, HIGH_DT_GRID_DELTA_T)

    mission_picm_l_1 = prob_survive_mission(samples, TARGET_DELTA_TS, mission_1, model_name="picm_l")["summary"]
    mission_picm_l_2 = prob_survive_mission(samples, TARGET_DELTA_TS, mission_2, model_name="picm_l")["summary"]
    mission_picm_c_1 = prob_survive_mission(picm_c_draws, TARGET_DELTA_TS, mission_1, model_name="picm_c", xbar=data.xbar)["summary"]
    mission_picm_c_2 = prob_survive_mission(picm_c_draws, TARGET_DELTA_TS, mission_2, model_name="picm_c", xbar=data.xbar)["summary"]
    mission_lognorm_1 = prob_survive_mission(lognorm_draws, TARGET_DELTA_TS, mission_1, model_name="lognormal")["summary"]
    mission_lognorm_2 = prob_survive_mission(lognorm_draws, TARGET_DELTA_TS, mission_2, model_name="lognormal")["summary"]

    table_low_modelwise = build_low_delta_modelwise_table(
        target_delta_ts=TARGET_DELTA_TS,
        pred_picm_l=pred_picm_l_targets,
        pred_picm_c=pred_picm_c_targets,
        pred_lognorm=pred_lognorm_targets,
        mission_picm_l_1=mission_picm_l_1,
        mission_picm_l_2=mission_picm_l_2,
        mission_picm_c_1=mission_picm_c_1,
        mission_picm_c_2=mission_picm_c_2,
        mission_lognorm_1=mission_lognorm_1,
        mission_lognorm_2=mission_lognorm_2,
    )
    table_low_envelope = build_low_delta_envelope_table(table_low_modelwise)
    table_low_modelwise.to_csv(TAB_DIR / "table_low_deltaT_modelwise.csv", index=False)
    table_low_envelope.to_csv(TAB_DIR / "table_low_deltaT_envelope.csv", index=False)

    mission_tables = {
        "PICM-L (Bayesian)": {mission_1: mission_picm_l_1, mission_2: mission_picm_l_2},
        "PICM-C (Laplace)": {mission_1: mission_picm_c_1, mission_2: mission_picm_c_2},
        "Lognormal AFT (Laplace)": {mission_1: mission_lognorm_1, mission_2: mission_lognorm_2},
    }
    table_mission_survival = build_mission_survival_table(mission_tables)
    table_mission_survival.to_csv(TAB_DIR / "table_mission_survival.csv", index=False)

    # 12) Use-condition B10 summaries for table D
    b10_mle = b10_delta_interval(mle["theta"], mle["cov"], DELTA_T_USE)
    b10_firth = b10_delta_interval(firth["theta"], firth["cov_pen"], DELTA_T_USE)
    b10_bayes_samples = post_pred["use_b10_samples"]
    b10_bayes = (
        float(np.median(b10_bayes_samples)),
        float(np.quantile(b10_bayes_samples, 0.025)),
        float(np.quantile(b10_bayes_samples, 0.975)),
    )
    b10_prof_unc = (
        float(np.exp(logpsi_hat)),
        float(profile["ci_unc_psi"][0]),
        float(profile["ci_unc_psi"][1]),
    )
    b10_prof_bc = (float(np.exp(logpsi_hat)), float(ci_bc[0]), float(ci_bc[1]))

    # Lognormal reference-case B10 for requested comparison
    sigma_ln, b1_ln, b0_ln = lognorm["theta"]
    mu_use_ln = b0_ln - b1_ln * np.log(DELTA_T_USE)
    lognormal_b10_use = float(np.exp(mu_use_ln + sigma_ln * stats.norm.ppf(0.1)))

    # 13) Tables B/C/D
    table_b = build_table_b(mle, firth, post_summary)
    table_c = build_table_c(mle, picm_c, lognorm, waic_dic, n_obs=data.n)
    table_d = build_table_d(
        delta_t_use=DELTA_T_USE,
        b10_mle=b10_mle,
        b10_firth=b10_firth,
        b10_bayes=b10_bayes,
        b10_prof_unc=b10_prof_unc,
        b10_prof_bc=b10_prof_bc,
        firth=firth,
    )
    table_b.to_csv(TAB_DIR / "table_B_picm_l_estimates.csv", index=False)
    table_c.to_csv(TAB_DIR / "table_C_model_comparison.csv", index=False)
    table_c.to_csv(TAB_DIR / "model_comparison_metrics.csv", index=False)
    table_d.to_csv(TAB_DIR / "table_D_use_condition_reliability.csv", index=False)

    table_model_comparison_summary = build_refreshed_model_comparison_table(
        picm_l_mle=mle,
        picm_c=picm_c,
        lognorm=lognorm,
        bayes_ic=waic_dic,
        n_obs=data.n,
    )
    table_model_comparison_summary.to_csv(TAB_DIR / "table_model_comparison.csv", index=False)

    table_group_shape = fit_group_specific_weibull_complete(
        data=data,
        pooled_picm_l_k=float(mle["theta"][0]),
        stress_values=(100.0, 125.0, 150.0),
        n_boot=N_GROUP_SHAPE_BOOT,
        seed=SEED_BOOT + 900,
    )
    table_group_shape_lrt = group_shape_common_k_lrt(data=data, stress_values=(100.0, 125.0, 150.0))
    table_group_shape_lrt.to_csv(TAB_DIR / "table_group_shape_lrt.csv", index=False)
    table_group_shape.to_csv(TAB_DIR / "table_group_shape_sensitivity.csv", index=False)
    plot_group_shape_sensitivity(table_group_shape, MAIN_FIG_DIR, lrt_table=table_group_shape_lrt)

    table_100c_units, table_100c_scatter = build_100c_scatter_tables(data=data, reference_max_hours=2678.0)
    table_100c_units.to_csv(TAB_DIR / "table_100C_observed_units.csv", index=False)
    table_100c_scatter.to_csv(TAB_DIR / "table_within_group_scatter_100C.csv", index=False)

    table_failure_only, table_failure_by_stress = compute_failure_only_regression_metrics(
        data=data,
        theta_picm_l=mle["theta"],
    )
    table_failure_only.to_csv(TAB_DIR / "table_failure_only_descriptive_metrics.csv", index=False)
    table_failure_by_stress.to_csv(TAB_DIR / "table_failure_only_rmse_by_stress.csv", index=False)

    table_high_dt = build_high_deltaT_extrapolation(theta_picm_l=mle["theta"], targets=HIGH_DT_TARGETS)
    table_high_dt.to_csv(TAB_DIR / "table_high_deltaT_extrapolation_175_200.csv", index=False)
    table_high_dt_posterior = build_high_deltaT_posterior_extrapolation(
        posterior_samples=samples,
        targets=HIGH_DT_TARGETS,
    )
    table_high_dt_posterior.to_csv(TAB_DIR / "table_high_deltaT_posterior_extrapolation_175_200.csv", index=False)

    table_low_duration = build_low_stress_duration(
        posterior_samples=samples,
        targets=LOW_STRESS_DURATION_TARGETS,
    )
    table_low_duration.to_csv(TAB_DIR / "table_low_stress_duration.csv", index=False)
    print("\nPosterior-median low-stress duration table:")
    print(table_low_duration.to_string(index=False))

    table_decision_crossings, table_decision_probabilities = build_decision_boundary_uncertainty(
        posterior_samples=samples,
        delta_grid=DECISION_PROB_DELTA_GRID,
        mission_targets=MISSION_CYCLE_TARGETS,
        probability_levels=DECISION_PROBABILITY_LEVELS,
    )
    table_decision_crossings.to_csv(TAB_DIR / "table_model_conditional_decision_boundary_uncertainty.csv", index=False)
    table_decision_probabilities.to_csv(TAB_DIR / "table_model_conditional_decision_probability_grid.csv", index=False)
    plot_decision_boundary_uncertainty(table_decision_probabilities, MAIN_FIG_DIR)

    acceptance_checks = build_acceptance_checks_v3p1(
        mle_theta=mle["theta"],
        mle_b10_35=b10_mle[0],
        firth_theta=firth["theta"],
        scatter_100c=table_100c_scatter,
        group_shape=table_group_shape,
        group_shape_lrt=table_group_shape_lrt,
        failure_metrics=table_failure_only,
        high_dt=table_high_dt,
        accept_rates=accept_rates,
        rhat=rhat,
        ess=ess,
        b10_bayes_35=b10_bayes[0],
        low_duration=table_low_duration,
    )
    acceptance_checks.to_csv(TAB_DIR / "table_v5_acceptance_checks.csv", index=False)
    print_and_assert_acceptance_checks(acceptance_checks)

    # Posterior predictive tables
    post_pred["stress_median"].to_csv(TAB_DIR / "posterior_stress_median_life.csv", index=False)
    post_pred["stress_b10"].to_csv(TAB_DIR / "posterior_stress_b10_life.csv", index=False)
    post_pred["use_summary"].to_csv(TAB_DIR / "posterior_use_condition_life_summary.csv", index=False)
    post_summary.to_csv(TAB_DIR / "posterior_parameter_summary.csv", index=False)

    # 14) Machine-readable MCMC diagnostics
    mcmc_diag = build_mcmc_diagnostics_table(accept_rates=accept_rates, rhat=rhat, ess=ess)
    mcmc_diag.to_csv(TAB_DIR / "mcmc_diagnostics.csv", index=False)

    # 15) Publication-quality figures
    generate_publication_figures(
        data=data,
        mle_theta=mle["theta"],
        firth_theta=firth["theta"],
        picm_c_theta=picm_c["theta"],
        bayes_samples=samples,
        pred_picm_l_grid=pred_picm_l_grid,
        pred_picm_c_grid=pred_picm_c_grid,
        pred_lognorm_grid=pred_lognorm_grid,
        pred_picm_l_high_grid=pred_picm_l_high_grid,
        pred_picm_l_targets=pred_picm_l_targets,
        pred_picm_c_targets=pred_picm_c_targets,
        pred_lognorm_targets=pred_lognorm_targets,
        table_high_dt_posterior=table_high_dt_posterior,
        chains=chains,
        rhat=rhat,
        ess=ess,
        accept_rates=accept_rates,
        post_pred=post_pred,
        profile_res=profile,
        bartlett_factor=bartlett_factor,
        b10_mle=b10_mle,
        b10_firth=b10_firth,
        b10_prof_unc=b10_prof_unc,
        b10_prof_bc=b10_prof_bc,
        failure_fraction_check=failure_fraction_check,
        loso_table=loso_table,
        prior_sensitivity_b10_35=prior_sens["b10_35_by_prior"],
        delta_t_use=DELTA_T_USE,
        main_dir=MAIN_FIG_DIR,
        supp_dir=SUPP_FIG_DIR,
        manifest_path=OUT_DIR / "figure_manifest.txt",
    )
    figure_manifest = finalise_figure_outputs()
    figure_qa_report = write_figure_qa_report(figure_manifest)
    print(f"Figure manifest rows: {len(figure_manifest)} -> {(FIG_DIR / 'figure_manifest.csv').resolve()}")
    print(f"Figure QA report -> {figure_qa_report.resolve()}")

    # 16) concise printed output + save key metrics
    print_summary(
        data=data,
        mle=mle,
        firth=firth,
        picm_c=picm_c,
        lognorm=lognorm,
        post_summary=post_summary,
        rhat=rhat,
        ess=ess,
        accept_rates=accept_rates,
        waic_dic=waic_dic,
        table_d=table_d,
        lognormal_b10_use=lognormal_b10_use,
    )
    narrative = write_analysis_results_summary(
        out_file=OUT_DIR / "analysis_results_summary.txt",
        modelwise_table=table_low_modelwise,
        envelope_table=table_low_envelope,
        mission_table=table_mission_survival,
    )
    print(narrative)

    metrics = pd.DataFrame(
        [
            {"metric": "WAIC", "value": waic_dic["waic"]},
            {"metric": "p_WAIC", "value": waic_dic["p_waic"]},
            {"metric": "DIC_supplementary", "value": waic_dic["dic"]},
            {"metric": "p_D", "value": waic_dic["p_dic"]},
            {"metric": "Bartlett_factor", "value": bartlett_factor},
            {"metric": "MCMC_accept_chain1", "value": float(accept_rates[0])},
            {"metric": "MCMC_accept_chain2", "value": float(accept_rates[1])},
            {"metric": "MCMC_accept_chain3", "value": float(accept_rates[2])},
            {"metric": "Rhat_k", "value": float(rhat[0])},
            {"metric": "Rhat_beta1", "value": float(rhat[1])},
            {"metric": "Rhat_beta0", "value": float(rhat[2])},
            {"metric": "ESS_k", "value": float(ess[0])},
            {"metric": "ESS_beta1", "value": float(ess[1])},
            {"metric": "ESS_beta0", "value": float(ess[2])},
            {"metric": "PICM_C_uncertainty_method", "value": picm_c_unc["method"]},
            {"metric": "Lognormal_uncertainty_method", "value": lognorm_unc["method"]},
            {"metric": "Mission_target_1", "value": mission_1},
            {"metric": "Mission_target_2", "value": mission_2},
        ]
    )
    metrics.to_csv(TAB_DIR / "key_metrics.csv", index=False)
    write_run_summary(
        out_file=OUT_DIR / "run_summary.txt",
        source_kind=source_kind,
        csv_path=csv_path,
        strict_validation_passed=validation_info.get("strict_pattern_passed"),
        waic_dic=waic_dic,
        accept_rates=accept_rates,
        rhat=rhat,
        ess=ess,
        delta_t_use=DELTA_T_USE,
        picm_c_uncertainty_method=str(picm_c_unc["method"]),
        lognormal_uncertainty_method=str(lognorm_unc["method"]),
    )

    result_payload = build_result_values_payload(
        delta_t_use=DELTA_T_USE,
        waic_dic=waic_dic,
        table_c=table_c,
        table_d=table_d,
        loso_table=loso_table,
        prior_sensitivity_table=prior_sens_summary,
        modelwise_table=table_low_modelwise,
    )
    write_result_values_files(
        payload=result_payload,
        txt_path=OUT_DIR / "analysis_values.txt",
        json_path=OUT_DIR / "analysis_values.json",
    )
    (OUT_DIR / "results_summary.json").write_text(json.dumps(result_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    package_dir = build_repeatability_package()
    print(f"Outputs written to: {OUT_DIR.resolve()}")
    print(f"Single-script repeatability package: {package_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()

