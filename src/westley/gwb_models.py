#!/usr/bin/env python3

from .fitter import BaseSplineModel, proposal, ProposalResult
import random
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d


class RedshiftSampler(BaseSplineModel):
    """
    Redshift Sampling Model:

    Be sure to run `set_base_population_information(pop_object,lambda_0)`
    before sampling. `pop_object` will be used to calculated omega_gw
    and lambda_0 are the stock set of parameters for the population model,
    many of which we need for the mass models, which we assume for now to be known.
    """

    def set_base_population_information(self, pop_object, lambda_0):
        self.pop_object = pop_object
        self.lambda_0 = lambda_0

    def ln_likelihood(self, config, heights, knots):
        # construct what needs to go into calculating omega_gw
        params = self.lambda_0
        params.update({**{f'amplitudes{ii}': heights[ii] for ii in range(heights.size)},
                       **{f'configuration{ii}': config[ii] for ii in range(config.size)},
                       **{f'xvals{ii}': knots[ii] for ii in range(self.available_knots.size)},
                       })
        self.pop_object.calculate_omega_gw(params)
        return np.sum(-0.5 * (self.data.data_yvals - self.pop_object.omega_gw)**2 / (2 * self.data.data_errors**2))

    @proposal(name='death', weight=1)
    def death(self):
        """Death proposal: Remove an existing knot."""
        active_idx = np.where(self.state.configuration)[0]
        if len(active_idx) <= self.min_knots:
            return None

        idx_to_remove = random.choices(active_idx, k=1)[0]
        new_config = self.state.configuration.copy()
        new_config[idx_to_remove] = False

        log_ratio = self._calculate_death_ratio(
            idx_to_remove, 
            self.state.heights,
            self.state.knots
        )

        new_ll = self.ln_likelihood(new_config, self.state.heights, self.state.knots)
        return ProposalResult(
            new_ll, log_ratio, new_config, 
            self.state.heights.copy(), self.state.knots.copy()
        )


class FitOmega(BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `SmoothCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """

    def evaluate_interp_model(self, x, heights, configuration, knots):
        active_knots = knots[configuration]
        active_heights = heights[configuration]
        if len(active_knots) == 0:
            return -40 * np.ones_like(x) # heights are log heights here...
        if len(active_knots) == 1:
            return active_heights[0]

        sorted_indices = np.argsort(active_knots)
        x_sorted = active_knots[sorted_indices]
        y_sorted = active_heights[sorted_indices]

        if self.interp_type == "linear":
            interpolator = interp1d(x_sorted, y_sorted, bounds_error=False, 
                                    fill_value='extrapolate')# (y_sorted[0], y_sorted[-1]))
        elif self.interp_type == "akima":
            interpolator = Akima1DInterpolator(x_sorted, y_sorted)
        else:
            raise ValueError(f"Unknown interpolation type: {self.interp_type}")

        return interpolator(x)

    def ln_likelihood(self, config, log10_heights, knots):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!

        # note that heights are being sampled in log space, but xvalues and knots are not
        # so we need to convert xvalues and knots to log space, do the interpolation in log space,
        # and then convert back to linear space
        omega_model = 10**self.evaluate_interp_model(
            np.log10(self.data.data_xvals),
            log10_heights,
            config,
            np.log10(knots))

        return np.sum(norm.logpdf(omega_model - self.data.data_yvals,
                                  scale=self.data.data_errors))
