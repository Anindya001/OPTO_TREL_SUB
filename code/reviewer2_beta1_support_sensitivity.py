#!/usr/bin/env python3
"""
Reviewer-2 support-sensitivity check for TR-2026-412.R2.

This check separates prior-scale sensitivity from prior-support sensitivity.
The baseline support beta1 > 0.5 is compared with beta1 > 0 while keeping
all other prior hyperparameters, likelihood terms, MCMC controls, starting
states, seeds, and data unchanged.

Important consequence
---------------------
For beta1 > 0.5, relaxing the truncation point changes the truncated-normal
prior only by a normalising constant. That constant cancels in Metropolis
ratios. Therefore, if the relaxed-support chain never enters the newly opened
region 0 < beta1 <= 0.5, a same-seed rerun should follow exactly the same
Markov-chain path as the baseline run (up to exceptional floating-point
tie-breaking). The script reports this directly rather than interpreting
ordinary Monte Carlo differences as support sensitivity.

Outputs
-------
tables/table_beta1_support_sensitivity.csv
tables/table_beta1_support_sensitivity_diagnostics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

import optocoupler_thermal_reliability_study_pubrev_v5 as core


REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = REPO_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def b10_draws(samples: np.ndarray, delta_t: float) -> np.ndarray:
    """Posterior draws of Weibull B10 life in cycles at the requested DeltaT."""
    k = samples[:, 0]
    beta1 = samples[:, 1]
    beta0 = samples[:, 2]
    eta = np.exp(beta0 - beta1 * np.log(float(delta_t)))
    return eta * ((-np.log(0.9)) ** (1.0 / k))


def run_setting(data, starts, cfg):
    """Run the same primary three-chain adaptive Metropolis calculation as v5."""
    fit = core.run_adaptive_mh_picm_l(
        data=data,
        starts=starts,
        seeds=core.SEED_MCMC,
        burnin=core.MCMC_BURNIN,
        keep=core.MCMC_KEEP,
        adapt_block=core.MCMC_ADAPT_BLOCK,
        prior_cfg=cfg,
    )
    chains = np.asarray(fit["chains"], dtype=float)
    samples = chains.reshape(-1, chains.shape[-1])
    rhat = core.compute_rhat(chains)
    ess = np.array(
        [core.effective_sample_size(chains[:, :, j]) for j in range(chains.shape[2])],
        dtype=float,
    )
    return fit, chains, samples, rhat, ess


def metric_summary(samples: np.ndarray, metric: str, delta_t: float | None = None):
    """Return median and 95% equal-tailed interval for beta1 or a B10 quantity."""
    if metric == "beta1":
        x = samples[:, 1]
    elif metric == "B10":
        if delta_t is None:
            raise ValueError("delta_t is required for B10")
        x = b10_draws(samples, delta_t)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    q025, q50, q975 = np.quantile(x, [0.025, 0.50, 0.975])
    return float(q50), float(q025), float(q975)


def main():
    df, _source, _csv_path = core.load_input_dataset(
        core.INPUT_CSV,
        core.CYCLE_TIME_HOURS,
    )
    core.validate_dataset(
        df,
        cycle_time_hours=core.CYCLE_TIME_HOURS,
        enforce_real_failure_pattern=core.ENFORCE_REAL_FAILURE_PATTERN,
        pattern_mismatch_is_error=core.REAL_PATTERN_MISMATCH_IS_ERROR,
    )
    data = core.prepare_data(df)

    mle_fit = core.fit_picm_l_mle(
        data=data,
        n_starts=core.N_OPT_STARTS_MAIN,
        seed=core.SEED_OPT,
    )
    mle_theta = np.asarray(mle_fit["theta"], dtype=float)

    baseline_cfg = core.get_prior_config(core.BASELINE_PRIOR_KEY)
    relaxed_cfg = dict(baseline_cfg)
    relaxed_cfg["beta1_trunc_lower"] = 0.0

    # Use the SAME starts as well as the same seeds.  This makes the
    # support-only comparison deterministic.
    common_starts = core.make_picm_l_mcmc_starts(mle_theta, prior_cfg=baseline_cfg)

    baseline_fit, baseline_chains, baseline_samples, baseline_rhat, baseline_ess = run_setting(
        data, common_starts, baseline_cfg
    )
    relaxed_fit, relaxed_chains, relaxed_samples, relaxed_rhat, relaxed_ess = run_setting(
        data, common_starts, relaxed_cfg
    )

    chains_identical = bool(np.array_equal(baseline_chains, relaxed_chains))
    max_abs_chain_diff = float(np.max(np.abs(baseline_chains - relaxed_chains)))
    relaxed_below_old_bound = float(np.mean(relaxed_samples[:, 1] <= 0.5))

    settings = {
        "baseline_beta1_gt_0p5": {
            "lower": 0.5,
            "samples": baseline_samples,
            "fit": baseline_fit,
            "rhat": baseline_rhat,
            "ess": baseline_ess,
        },
        "support_relaxed_beta1_gt_0": {
            "lower": 0.0,
            "samples": relaxed_samples,
            "fit": relaxed_fit,
            "rhat": relaxed_rhat,
            "ess": relaxed_ess,
        },
    }

    baseline_reference = {}
    for metric, dt in [("beta1", None), ("B10", 20.0), ("B10", 25.0), ("B10", 35.0)]:
        key = "beta1" if metric == "beta1" else f"B10_{int(dt)}C_cycles"
        baseline_reference[key] = metric_summary(baseline_samples, metric, dt)

    rows = []
    for setting_name, obj in settings.items():
        samples = obj["samples"]
        for metric, dt in [("beta1", None), ("B10", 20.0), ("B10", 25.0), ("B10", 35.0)]:
            metric_name = "beta1" if metric == "beta1" else f"B10_{int(dt)}C_cycles"
            median, lo, hi = metric_summary(samples, metric, dt)
            base_median = baseline_reference[metric_name][0]
            shift = 100.0 * (median / base_median - 1.0)
            rows.append(
                {
                    "setting": setting_name,
                    "beta1_trunc_lower": obj["lower"],
                    "metric": metric_name,
                    "median": median,
                    "eti_2.5": lo,
                    "eti_97.5": hi,
                    "relative_median_shift_percent_vs_baseline": shift,
                }
            )

    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "table_beta1_support_sensitivity.csv", index=False
    )

    diag_rows = []
    parameter_names = ["k", "beta1", "beta0"]
    for setting_name, obj in settings.items():
        for chain_no, acc in enumerate(obj["fit"]["accept_rates"], start=1):
            diag_rows.append(
                {
                    "setting": setting_name,
                    "diagnostic": "chain_acceptance",
                    "parameter_or_chain": f"chain_{chain_no}",
                    "value": float(acc),
                }
            )
        for j, name in enumerate(parameter_names):
            diag_rows.append(
                {
                    "setting": setting_name,
                    "diagnostic": "rhat",
                    "parameter_or_chain": name,
                    "value": float(obj["rhat"][j]),
                }
            )
            diag_rows.append(
                {
                    "setting": setting_name,
                    "diagnostic": "ess",
                    "parameter_or_chain": name,
                    "value": float(obj["ess"][j]),
                }
            )

    diag_rows.extend(
        [
            {
                "setting": "support_relaxed_beta1_gt_0",
                "diagnostic": "posterior_fraction_beta1_le_0p5",
                "parameter_or_chain": "beta1",
                "value": relaxed_below_old_bound,
            },
            {
                "setting": "support_comparison",
                "diagnostic": "chains_identical_same_seed",
                "parameter_or_chain": "all",
                "value": 1.0 if chains_identical else 0.0,
            },
            {
                "setting": "support_comparison",
                "diagnostic": "max_abs_chain_difference",
                "parameter_or_chain": "all",
                "value": max_abs_chain_diff,
            },
        ]
    )

    pd.DataFrame(diag_rows).to_csv(
        TABLE_DIR / "table_beta1_support_sensitivity_diagnostics.csv", index=False
    )

    print("\nReviewer-2 support-sensitivity check")
    print("====================================")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSame-seed comparison:")
    print("chains identical:", chains_identical)
    print("maximum absolute chain difference:", max_abs_chain_diff)
    print("fraction of relaxed posterior draws with beta1 <= 0.5:", relaxed_below_old_bound)


if __name__ == "__main__":
    main()
