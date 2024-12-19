#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

from westley import fitter
from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# class TestBaseSplineModel(fitter.BaseSplineModel):
#     def ln_likelihood(self, config, heights, knot_locations):
#         yvals_model = self.evaluate_interp_model(knot_locations, heights, config)
#         return -0.5 * np.sum((self.data.yvals - yvals_model)**2 / self.data.data_errors**2)

class TestFitter:
    # fake data for fitting
    fake_xvals = np.arange(5)
    fake_yval_errors = 0.1 * np.ones(5)
    fake_noise = np.random.randn(5) * 0.1

    # fake data with noise
    fake_yvals = np.sin(2 * np.pi * fake_xvals / 5) + fake_noise

    # fake data object
    fake_data = fitter.SmoothCurveDataObj(fake_xvals,
                                          fake_yvals,
                                          fake_yval_errors)

    bsm = fitter.FitSmoothCurveModel(fake_data,
                                 5,
                                 (0, 5),
                                 (-2, 2),
                                 interp_type='linear',
                                 log_output=False,
                                 log_space_xvals=False,
                                 birth_uniform_frac=0.5,
                                 min_knots=2,
                                 birth_gauss_scalefac=1
                                    )


    def test_prior_run_full_proposal(self):

        results = self.bsm.sample(Niterations=50000,
                                  prior_test=True)

        # thin by factor of 100 because
        # of possible correlations between samples
        num_knots = results.get_num_knots()[::100]
        p_value = check_uniform_distribution(num_knots, bins=10, discrete=True)
        assert p_value > 0.05, f"Distribution of knot counts is not uniform (p={p_value:.3f})"

    def test_prior_run_amplitude_prior_proposal(self):

        results = self.bsm.sample(Niterations=50000,
                                  prior_test=True,
                                  proposal_cycle={'change_amplitude_prior_draw': 1})

        # thin by factor of 100 because
        # of possible correlations between samples
        p_value = check_uniform_distribution(results.heights[::100, 0],
                                             range=(self.bsm.ylow,
                                                    self.bsm.yhigh - self.bsm.ylow))
        plt.hist(results.heights[::100, 0])
        plt.savefig("height_samples.png")
        np.savetxt("height_samples.txt", results.heights[::100, 0])
        assert p_value > 0.05, f"Distribution of knot counts is not uniform (p={p_value:.3f})"


def check_uniform_distribution(arr, bins=10, discrete=False, range=None):
    # for the discrete case, we need to use the unique values
    if discrete:
        bins, counts = np.unique(arr, return_counts=True)
        expected_counts = arr.size / (bins.size)
        # counting error
        expected_count_error = np.sqrt(expected_counts)
        # Gaussian uncertainty (as long as counts are large)
        pval_per_bin = 1 - norm.cdf(counts, loc=expected_counts, scale=expected_count_error)
        # check that p-values are uniform across bins
        ks_stat, p_value = kstest(pval_per_bin, 'uniform')
    # for the continuous case, we need to use the histogram
    else:
        ks_stat, p_value = kstest(arr, 'uniform', args=(range[0], range[1]))

    return p_value

