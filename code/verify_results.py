"""Fast verification of the retained numerical results.

The script independently recomputes the principal maximum-likelihood results
from the unit-level data and checks posterior summaries directly from the
retained draws.  It exits with a non-zero status if any required check fails.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import (
    DRAWS_DIR,
    RESULTS_DIR,
    common_shape_lrt,
    fit_primary_mle,
    load_dataset,
    omission_subsets,
    profile_b10_interval,
    weibull_b10,
    write_json,
)

OUTPUT = RESULTS_DIR / "verification_summary.json"
checks: list[dict[str, object]] = []


def check(name: str, actual: float, expected: float, tolerance: float) -> None:
    passed = math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
    checks.append(
        {
            "check": name,
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "status": "PASS" if passed else "FAIL",
        }
    )


def check_exact(name: str, actual: object, expected: object) -> None:
    passed = actual == expected
    checks.append(
        {
            "check": name,
            "actual": actual,
            "expected": expected,
            "tolerance": 0,
            "status": "PASS" if passed else "FAIL",
        }
    )


def main() -> int:
    data = load_dataset()

    check_exact("device count", len(data), 48)
    check_exact("failure count", int(data["event"].sum()), 29)
    check_exact("run-out count", int((1 - data["event"]).sum()), 19)

    expected_failures = {25: 0, 50: 0, 75: 5, 100: 8, 125: 8, 150: 8}
    observed_failures = (
        data.groupby("delta_T_C")["event"].sum().astype(int).to_dict()
    )
    check_exact("failures by excursion", observed_failures, expected_failures)

    expected_omission = pd.read_csv(RESULTS_DIR / "omission.csv").set_index("analysis")
    for name, subset in omission_subsets(data):
        fit = fit_primary_mle(subset)
        theta = np.asarray(fit["theta"], dtype=float)
        expected = expected_omission.loc[name]
        check(f"{name}: k", theta[0], expected["k"], 2.0e-5)
        check(f"{name}: beta1", theta[1], expected["beta1"], 2.0e-5)
        check(f"{name}: beta0", theta[2], expected["beta0"], 5.0e-5)
        check(
            f"{name}: B10(25)",
            float(weibull_b10(theta, 25.0)),
            expected["B10_25"],
            0.2,
        )

    expected_profile = json.loads(
        (RESULTS_DIR / "profile_35.json").read_text(encoding="utf-8")
    )
    profile = profile_b10_interval(data, 35.0)
    for key in ("mle", "lower", "upper"):
        check(f"profile B10(35): {key}", profile[key], expected_profile[key], 0.2)

    group_fits, common_shape = common_shape_lrt(data)
    expected_groups = pd.read_csv(RESULTS_DIR / "group_shape_fits.csv")
    for actual, expected in zip(
        group_fits.itertuples(index=False), expected_groups.itertuples(index=False)
    ):
        check(f"shape at {actual.delta_T:.0f} C", actual.k, expected.k, 1.0e-5)
        check(
            f"scale at {actual.delta_T:.0f} C",
            actual.eta_cycles,
            expected.eta_cycles,
            1.0e-4,
        )
    expected_common = pd.read_csv(RESULTS_DIR / "common_shape.csv").iloc[0]
    check("common-shape LRT", common_shape["lr_stat"], expected_common.lr_stat, 1.0e-5)
    check("common-shape p-value", common_shape["p_value"], expected_common.p_value, 1.0e-6)

    baseline = np.load(DRAWS_DIR / "baseline_chains.npy").reshape(-1, 3)
    expected_life = pd.read_csv(RESULTS_DIR / "primary_life.csv").set_index("delta_T")
    for excursion in (20.0, 25.0, 35.0):
        values = weibull_b10(baseline, excursion)
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        expected = expected_life.loc[excursion]
        check(f"B10({excursion:g}) q025", q025, expected.q025, 1.0e-8)
        check(f"B10({excursion:g}) median", median, expected["median"], 1.0e-8)
        check(f"B10({excursion:g}) q975", q975, expected.q975, 1.0e-8)

    curved = np.load(DRAWS_DIR / "curved_laplace.npy")
    xbar = np.mean(np.log([25.0, 50.0, 75.0, 100.0, 125.0, 150.0]))
    derivative_25 = -curved[:, 1] + 2.0 * curved[:, 3] * (np.log(25.0) - xbar)
    derivative_150 = -curved[:, 1] + 2.0 * curved[:, 3] * (np.log(150.0) - xbar)
    monotone_25_150 = (derivative_25 <= 0.0) & (derivative_150 <= 0.0)
    check(
        "PICM-C non-monotone fraction over 25-150 C",
        1.0 - float(np.mean(monotone_25_150)),
        0.1747,
        5.0e-5,
    )

    failures = [item for item in checks if item["status"] != "PASS"]
    write_json(
        OUTPUT,
        {
            "checks": checks,
            "passed": len(checks) - len(failures),
            "failed": len(failures),
        },
    )

    for item in checks:
        print(f"{item['status']:4s}  {item['check']}")
    print(f"\n{len(checks) - len(failures)} passed; {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
