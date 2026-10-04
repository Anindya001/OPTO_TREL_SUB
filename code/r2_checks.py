"""Reproduce the R2 omission, support and monotonicity checks.

All fits use the unit-level data with cycles calculated as hours / 3.
The intercept is profiled analytically in the omission fits. Retained draws
are used for the prior-support and comparator-conditioning summaries.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from analysis_core import DRAWS_DIR, RESULTS_DIR, load_dataset, weibull_b10


def profiled_fit(data):
    """Maximise the exact Weibull likelihood after profiling the intercept."""
    x = np.log(data['delta_T_C'].to_numpy(float))
    log_y = np.log(data['cycles'].to_numpy(float))
    event = data['event'].to_numpy(float)
    failures = event.sum()
    sy, sx = event @ log_y, event @ x

    def objective(theta):
        k, slope = theta
        v = log_y + slope * x
        log_total = logsumexp(k * v)
        weight = np.exp(k * v - log_total)
        value = (failures * np.log(k) + (k - 1) * sy + k * slope * sx
                 - failures * (log_total - np.log(failures)) - failures)
        gradient = np.array([failures / k + sy + slope * sx - failures * (weight @ v),
                             k * (sx - failures * (weight @ x))])
        return -value, -gradient

    candidates = [minimize(objective, start, jac=True, method='L-BFGS-B',
                           bounds=[(1.0000001, 10), (0.000001, 10)],
                           options={'ftol': 1e-14, 'gtol': 1e-9, 'maxiter': 10000})
                  for start in [(3, 2.4), (2, 1), (4, 3)]]
    fit = min(candidates, key=lambda result: result.fun)
    k, slope = fit.x
    intercept = (logsumexp(k * (log_y + slope * x)) - np.log(failures)) / k
    gradient = np.linalg.norm(objective(fit.x)[1], ord=np.inf)
    if not np.isfinite(fit.fun) or gradient > 1e-5:
        raise RuntimeError(f'Omission fit did not converge: gradient={gradient:g}')
    return np.array([k, slope, intercept]), float(-fit.fun), float(gradient)


def omission_results(data):
    low = data['delta_T_C'].isin([25, 50])
    transition = data['delta_T_C'] == 75
    runout = data['event'] == 0
    subsets = [('All observations', data),
               ('Omit 25/50 C run-outs', data[~(low & runout)]),
               ('Omit 75 C run-outs', data[~(transition & runout)]),
               ('Omit all run-outs', data[~runout]),
               ('Omit whole 75 C group', data[~transition]),
               ('Omit 75 C failures', data[~(transition & ~runout)]),
               ('Only 100-150 C groups', data[data['delta_T_C'] >= 100])]
    rows = []
    for name, subset in subsets:
        theta, loglik, gradient = profiled_fit(subset)
        rows.append(dict(analysis=name, n=len(subset), k=theta[0], beta1=theta[1],
                         beta0=theta[2], B10_25=float(weibull_b10(theta, 25)),
                         loglik=loglik, gradient_max=gradient))
    result = pd.DataFrame(rows)
    result['change_percent'] = 100 * (result.B10_25 / result.B10_25.iloc[0] - 1)
    return result


def conditioning_results():
    draws = np.load(DRAWS_DIR / 'curved_laplace.npy')
    k, slope, intercept, curvature = draws.T
    xbar = np.log([25, 50, 75, 100, 125, 150]).mean()
    masks = {'Unconditioned': np.ones(len(draws), dtype=bool)}
    for low in [25, 20]:
        derivative_low = -slope + 2 * curvature * (np.log(low) - xbar)
        derivative_high = -slope + 2 * curvature * (np.log(150) - xbar)
        masks[f'Monotone {low}-150 C'] = (derivative_low <= 0) & (derivative_high <= 0)
    rows = []
    for name, mask in masks.items():
        for stress in [20, 25, 35]:
            b10 = np.exp(intercept - slope * np.log(stress)
                         + curvature * (np.log(stress) - xbar) ** 2
                         + np.log(-np.log(0.9)) / k)
            lower, median, upper = np.quantile(b10[mask], [.025, .5, .975])
            rows.append(dict(condition=name, delta_T=stress,
                             removed_fraction=1-float(mask.mean()),
                             q025=lower, median=median, q975=upper))
    return pd.DataFrame(rows)


def support_results():
    rows = []
    for name in ['baseline', 'relaxed', 'diffuse', 'conservative']:
        draws = np.load(DRAWS_DIR / f'{name}_chains.npy').reshape(-1, 3)
        for stress in [20, 25, 35]:
            lower, median, upper = np.quantile(weibull_b10(draws, stress), [.025, .5, .975])
            rows.append(dict(setting=name, delta_T=stress, n_draws=len(draws),
                             beta1_at_or_below_0p5=int(np.count_nonzero(draws[:, 1] <= .5)),
                             q025=lower, median=median, q975=upper))
    return pd.DataFrame(rows)


def main():
    out = RESULTS_DIR / 'r2_checks'
    out.mkdir(exist_ok=True)
    omission = omission_results(load_dataset())
    conditioning = conditioning_results()
    support = support_results()
    omission.to_csv(out / 'omission.csv', index=False)
    conditioning.to_csv(out / 'monotonicity.csv', index=False)
    support.to_csv(out / 'prior_support.csv', index=False)
    whole = omission.set_index('analysis').loc['Omit whole 75 C group']
    assert abs(whole.B10_25 - 27637.8) < .1
    assert abs(whole.change_percent - 218.59) < .01
    conditioned20 = conditioning[(conditioning.condition == 'Monotone 20-150 C')
                                 & (conditioning.delta_T == 20)].iloc[0]
    assert abs(conditioned20.q025 - 1802.2) < .1
    assert abs(conditioned20.removed_fraction - .2346) < .0001
    print(omission.to_string(index=False))
    print(f'PICM-C conditioned lower endpoint at 20 C: {conditioned20.q025:.1f}')
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
