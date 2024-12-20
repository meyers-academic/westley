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

    def test_model_initialization(self):
        """Test model initialization and properties"""
        assert self.bsm.N_possible_knots == 5
        assert self.bsm.min_knots == 2
        assert self.bsm.ylow == -2
        assert self.bsm.yhigh == 2
        assert self.bsm.yrange == 4
        assert len(self.bsm.available_knots) == 5

    def test_evaluate_interp_model_single_knot(self):
        """Test interpolation with single knot"""
        test_config = np.zeros(5, dtype=bool)
        test_config[0] = True
        test_heights = np.ones(5)
        result = self.bsm.evaluate_interp_model(np.array([2.5]), test_heights, test_config, self.bsm.available_knots)
        assert np.allclose(result, 1.0)

    def test_evaluate_interp_model_multiple_knots(self):
        """Test interpolation with multiple knots"""
        test_config = np.ones(5, dtype=bool)
        test_heights = np.ones(5)
        x_test = np.linspace(0, 5, 10)
        result = self.bsm.evaluate_interp_model(x_test, test_heights, test_config, self.bsm.available_knots)
        assert len(result) == len(x_test)
        assert np.allclose(result, 1.0)

    def test_birth_proposal_mechanics(self):
        """Test birth proposal basics"""
        self.bsm.configuration = np.zeros(5, dtype=bool)
        self.bsm.configuration[0:2] = True  # Start with minimum knots
        
        ll, logR, config, heights, knots = self.bsm.birth()
        
        if ll != -np.inf:  # If proposal was valid
            assert np.sum(config) == 3  # Should have one more knot
            assert heights.shape == (5,)
            assert knots.shape == (5,)

    def test_death_proposal_mechanics(self):
        """Test death proposal basics"""
        self.bsm.configuration = np.ones(5, dtype=bool)
        
        ll, logR, config, heights, knots = self.bsm.death()
        
        if ll != -np.inf:  # If proposal was valid
            assert np.sum(config) == 4  # Should have one less knot
            assert heights.shape == (5,)
            assert knots.shape == (5,)

    def test_change_amplitude_gaussian_mechanics(self):
        """Test Gaussian amplitude change proposal"""
        original_heights = self.bsm.current_heights.copy()
        
        ll, logR, config, heights, knots = self.bsm.change_amplitude_gaussian()
        
        if ll != -np.inf:
            # Only one height should change
            diff = heights != original_heights
            assert np.sum(diff) == 1
            # Changed height should be within bounds
            assert np.all(heights >= self.bsm.ylow)
            assert np.all(heights <= self.bsm.yhigh)

    def test_change_amplitude_prior_draw_mechanics(self):
        """Test prior draw amplitude change proposal"""
        original_heights = self.bsm.current_heights.copy()
        
        ll, logR, config, heights, knots = self.bsm.change_amplitude_prior_draw()
        
        if ll != -np.inf:
            # Only one height should change
            diff = heights != original_heights
            assert np.sum(diff) == 1
            # New height should be within prior bounds
            assert np.all(heights >= self.bsm.ylow)
            assert np.all(heights <= self.bsm.yhigh)

    def test_change_knot_location_mechanics(self):
        """Test knot location change proposal"""
        original_knots = self.bsm.available_knots.copy()
        
        ll, logR, config, heights, knots = self.bsm.change_knot_location()
        
        if ll != -np.inf:
            # Only one knot should change
            diff = knots != original_knots
            assert np.sum(diff) == 1
            # New knot should be within bounds
            idx_changed = np.where(diff)[0][0]
            assert knots[idx_changed] >= self.bsm.xlows[idx_changed]
            assert knots[idx_changed] <= self.bsm.xhighs[idx_changed]

    def test_height_prior_bounds(self):
        """Test height prior boundary conditions"""
        # Test below lower bound
        assert self.bsm.get_height_log_prior(self.bsm.ylow - 1) == -np.inf
        # Test above upper bound
        assert self.bsm.get_height_log_prior(self.bsm.yhigh + 1) == -np.inf
        # Test valid height
        valid_height = (self.bsm.yhigh + self.bsm.ylow) / 2
        assert self.bsm.get_height_log_prior(valid_height) == -np.log(self.bsm.yrange)

    def test_amplitude_prior_draw_proposal(self):
        """Test that amplitude prior draw actually changes values"""
        initial_height = self.bsm.current_heights[0]
        
        # Perform several proposals
        different_values = set()
        for _ in range(100):
            new_ll, ratio, new_config, new_heights, new_knots = self.bsm.change_amplitude_prior_draw()
            different_values.add(new_heights[0])
        
        # Should get multiple different values
        assert len(different_values) > 1, "Amplitude proposals not generating different values"

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
