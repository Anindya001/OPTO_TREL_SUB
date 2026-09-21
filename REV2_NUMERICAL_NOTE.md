# TR-2026-412.R2 numerical verification for Reviewer 2

This note documents the additional numerical work requested in the second major revision.

## Scope of new calculation

Reviewer 2 correctly observed that the existing baseline/diffuse/conservative prior sweep changed prior scales while retaining the same support restriction `beta1 > 0.5`. That analysis therefore tested prior-scale sensitivity, not prior-support sensitivity.

The support-sensitivity check here repeats the primary Bayesian Weibull log-linear analysis with only one change:

- baseline: `beta1 > 0.5`
- relaxed support: `beta1 > 0`

All other likelihood terms, priors, MCMC settings, seeds, and the 48-unit dataset are unchanged.

## Reproducibility check

The baseline rerun reproduces the archived Revision-1 posterior exactly to numerical precision:

- beta1 posterior median: 2.458146
- 95% ETI: [1.917221, 3.027316]
- chain acceptance: 0.21515, 0.23925, 0.24805
- R-hat: all approximately 1.0001
- minimum ESS: approximately 3828

## Relaxed-support result

With the lower support relaxed from 0.5 to 0:

- beta1 posterior median = 2.464994
- 95% ETI = [1.908766, 3.027768]
- posterior median shift relative to baseline = +0.279%

B10 posterior medians change only modestly:

| DeltaT | Baseline B10 | Relaxed-support B10 | Relative shift |
|---|---:|---:|---:|
| 20 C | 15187.6 cycles | 15343.1 cycles | +1.024% |
| 25 C | 8772.8 cycles | 8837.6 cycles | +0.739% |
| 35 C | 3845.8 cycles | 3861.1 cycles | +0.399% |

The largest median shift among these reported low/near-boundary quantities is about 1.02%.

No retained relaxed-support posterior draw had `beta1 <= 0.5`. Thus, in this dataset, the former lower bound is inactive over the sampled posterior mass.

## MCMC diagnostics for relaxed support

- chain acceptance: 0.22170, 0.23925, 0.24805
- R-hat [k, beta1, beta0]: 1.000548, 1.000063, 1.000065
- ESS [k, beta1, beta0]: 4554.8, 3886.8, 3590.2

These satisfy the manuscript's convergence criteria.

## Reviewer-response interpretation

The correct conclusion is narrow:

> The earlier diffuse/conservative analysis tests sensitivity to prior scale within fixed support. A separate rerun relaxing the stress-slope support from beta1 > 0.5 to beta1 > 0 produces essentially unchanged posterior inference: beta1 shifts by about 0.28%, the largest reported B10 median shift is about 1.02%, and no retained posterior draw enters the previously excluded beta1 <= 0.5 region.

This supports the statement that the lower support restriction is not materially driving the present posterior. It is not a claim of robustness to arbitrary prior families or alternative stress-life models.

## Related verification already present in repository

The archived 100 C group-specific Weibull table reports:

- observed range: 720–2818 h
- fitted 5–95% individual-lifetime range: 722.465–3102.594 h
- fitted 2.5–97.5% individual-lifetime range: 560.985–3342.811 h

These are fitted individual-lifetime percentile ranges, not confidence intervals for the fitted median-life curve.
