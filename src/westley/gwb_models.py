#!/usr/bin/env python3

from .fitter import BaseSplineModel
import numpy as np


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


class FitOmega(BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `SmoothCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """

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
            np.log10(self.data.data_xvals), log10_heights, config, np.log10(knots))

        return np.sum(norm.logpdf(omega_model - self.data.data_yvals, scale=self.data.data_errors))
