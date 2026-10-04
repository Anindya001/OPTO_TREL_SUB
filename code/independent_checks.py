"""Independent posterior integration for Bayesian omissions and LOSO counts.

Gauss-Legendre quadrature integrates shape, slope and a transformed intercept.
For each shape/slope pair, t = k*a - log(H/D), where a is the intercept
centred on the observed log-stress mean, H = sum(exp(k*(log(y)+b*x_c)))
and D is the failure count. This centres the narrow conditional-intercept
likelihood at t=0 without changing the prior or likelihood. The Jacobian
is da/dt=1/k. Quantile CDFs integrate to their exact t cutoffs rather than
sorting a discrete cloud of integration nodes.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'optocoupler_ttf_unit_level.csv'
RESULTS_DIR = ROOT / 'results'


def nodes_weights(n, lower, upper):
    nodes, weight = leggauss(n)
    return lower + (nodes+1)*(upper-lower)/2, weight*(upper-lower)/2


class PosteriorIntegral:
    def __init__(self, data, nodes=80, wide=False):
        self.bounds = {'k': [1., 12. if wide else 10.],
                       'slope': [.5, 9. if wide else 7.5],
                       't': [-3., 4.] if wide else [-2., 3.]}
        kk, kw = nodes_weights(nodes, *self.bounds['k'])
        bb, bw = nodes_weights(nodes, *self.bounds['slope'])
        self.k, self.b = np.meshgrid(kk, bb, indexing='ij')
        self.log_kb_weight = np.log(kw[:, None] * bw[None, :])
        self.base_nodes, self.base_weights = leggauss(max(nodes, 100))
        x = np.log(data.delta_T_C.to_numpy(float))
        y = np.log(data.cycles.to_numpy(float))
        event = data.event.to_numpy(float)
        self.xbar = float(x.mean())
        xc = x-self.xbar
        self.d = float(event.sum())
        logh = logsumexp(self.k[..., None] * (y + self.b[..., None]*xc), axis=2)
        self.log_h_over_d = logh - np.log(self.d)
        self.constant = (self.d*np.log(self.k) + (self.k-1)*(event@y)
                         + self.k*self.b*(event@xc) - self.d*self.log_h_over_d
                         + stats.gamma.logpdf(self.k, 9, scale=1/3)
                         + stats.norm.logpdf(self.b, 2.5, .8) - np.log(self.k)
                         + self.log_kb_weight)
        self.t, self.log_weights = self.integrand(self.bounds['t'][1])
        self.log_norm = float(logsumexp(self.log_weights))
        self.weight = np.exp(self.log_weights-self.log_norm)

    def integrand(self, upper):
        lower = self.bounds['t'][0]
        upper = np.clip(upper, lower, self.bounds['t'][1])
        half = np.broadcast_to((upper-lower)/2, self.k.shape)
        t = lower + (self.base_nodes+1)*half[..., None]
        a = (self.log_h_over_d[..., None]+t)/self.k[..., None]
        intercept = a+self.b[..., None]*self.xbar
        with np.errstate(divide='ignore'):
            logwt = (self.constant[..., None] - self.d*t - self.d*np.exp(-t)
                     + stats.norm.logpdf(intercept, 18, 4)
                     + np.log(half[..., None]*self.base_weights))
        return t, logwt

    def b10_cdf(self, log_target, stress=25):
        cutoff = (self.k*(log_target-self.b*(self.xbar-np.log(stress)))
                  - np.log(-np.log(.9))-self.log_h_over_d)
        _, logwt = self.integrand(cutoff)
        return float(np.exp(logsumexp(logwt)-self.log_norm))

    def b10_median(self, stress=25):
        value = brentq(lambda z: self.b10_cdf(z, stress)-.5,
                       np.log(1.), np.log(1e9), xtol=1e-11)
        return float(np.exp(value))

    def counts(self, stress, horizon, n=8):
        a = (self.log_h_over_d[..., None]+self.t)/self.k[..., None]
        logscale = a-self.b[..., None]*(np.log(stress)-self.xbar)
        hazard = np.exp(np.clip(self.k[..., None]*(np.log(horizon)-logscale), -700, 700))
        probability = -np.expm1(-hazard)
        return np.array([np.sum(self.weight*stats.binom.pmf(count,n,probability))
                         for count in range(n+1)])


def run(nodes, wide):
    data = pd.read_csv(DATA_FILE)
    data['cycles'] = data.ttf_hours/3
    runout = data.event==0
    medians = {}
    for name, subset in [('full',data),
                         ('omit_25_50_runouts',data[~(data.delta_T_C.isin([25,50]) & runout)]),
                         ('omit_75_runouts',data[~((data.delta_T_C==75) & runout)])]:
        integral = PosteriorIntegral(subset,nodes,wide)
        medians[name] = integral.b10_median()
    medians['change_omit_25_50_percent'] = 100*(medians['omit_25_50_runouts']/medians['full']-1)
    medians['change_omit_75_percent'] = 100*(medians['omit_75_runouts']/medians['full']-1)
    loso=[]
    for stress in [25,50,75,100,125,150]:
        observed=data[data.delta_T_C==stress]
        horizon=float(observed.cycles.max())
        integral=PosteriorIntegral(data[data.delta_T_C!=stress],nodes,wide)
        pmf=integral.counts(stress,horizon)
        cdf=np.cumsum(pmf)
        count=int(observed.event.sum())
        loso.append(dict(held_out_stress=stress,horizon_cycles=horizon,
                         observed_failures=count,median=int(np.searchsorted(cdf,.5)),
                         lower_95=int(np.searchsorted(cdf,.025)),upper_95=int(np.searchsorted(cdf,.975)),
                         probability_at_least_observed=float(pmf[count:].sum()),pmf=pmf.tolist()))
    return dict(method='Gauss-Legendre integration with transformed intercept',
                nodes_per_shape_and_slope=nodes,intercept_nodes=max(nodes,100),
                bounds=integral.bounds,bayesian_omission_B10_25=medians,loso=loso)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,default=160)
    parser.add_argument('--wide',action='store_true')
    args=parser.parse_args()
    result=run(args.nodes,args.wide)
    path=RESULTS_DIR/('independent_grid_check_wide.json' if args.wide else 'independent_grid_check.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
