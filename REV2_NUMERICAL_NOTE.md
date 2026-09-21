# TR-2026-412.R2 numerical verification for Reviewer 2

## Correction to the support-sensitivity interpretation

Reviewer 2 correctly observed that the earlier baseline/diffuse/conservative sweep
changed prior scale while retaining the lower support beta1 > 0.5.  That analysis
therefore tested prior-scale sensitivity, not prior-support sensitivity.

The requested support check changes only the truncation point:

- baseline: beta1 > 0.5
- relaxed support: beta1 > 0

All other prior hyperparameters, likelihood terms, starting states, MCMC controls,
seeds, and the 48-unit dataset are held fixed.

For beta1 > 0.5, relaxing the lower truncation changes the truncated-normal prior only
by a normalising constant.  That constant cancels from Metropolis acceptance ratios.
Accordingly, if the relaxed chain never enters the newly opened interval
0 < beta1 <= 0.5, the same-seed Markov-chain path is unchanged.

The deterministic support-only rerun gives exactly that result:

- baseline and relaxed-support chains are identical under the same starts and seeds;
- maximum absolute difference between the retained chains is zero;
- no retained relaxed-support draw has beta1 <= 0.5.

Thus the baseline posterior summaries are unchanged:

- beta1 posterior median: 2.458146
- 95% ETI: [1.917221, 3.027316]
- B10(20 C) posterior median: 15187.6 cycles
- B10(25 C) posterior median: 8772.8 cycles
- B10(35 C) posterior median: 3845.8 cycles

The appropriate reviewer-response conclusion is therefore stronger and simpler:

> Relaxing the lower support from beta1 > 0.5 to beta1 > 0 leaves the same-seed
> posterior chain unchanged, and the relaxed-support posterior places no sampled mass
> in the previously excluded region.  The beta1 > 0.5 support restriction is therefore
> not active for this dataset.

This statement is limited to the support restriction itself.  It does not imply
robustness to arbitrary prior scales, alternative prior families, or alternative
stress-life model forms.

## Related 100 C verification

The archived group-specific Weibull result remains:

- observed range: 720-2818 h
- fitted 5-95% individual-lifetime range: 722.465-3102.594 h
- fitted 2.5-97.5% individual-lifetime range: 560.985-3342.811 h

These are fitted individual-lifetime percentile ranges, not confidence intervals for
the fitted median-life relation.
