#!/usr/bin/env python3
#
import gwpopulation
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator
from popstock.PopulationOmegaGW import PopulationOmegaGW

plpp_md_default_params = {'alpha': 2.5, 'beta': 1, 'delta_m': 3, 'lam': 0.04, 'mmax': 100,
                          'mmin': 4, 'mpp': 33, 'sigpp':5,
                          'gamma': 2.7, 'kappa': 3, 'z_peak': 1.9, 'rate': 15}


def create_initialize_popstock_model(freqs, mass_model, redshift_model,
                                     initial_params,
                                     num_proposal_samples=10000):
    """create and initialize a popstock populatio model
    for omega_gw using a user-supplied mass model and redshift model.
    will automatically initialize to the user-supplied initial parameters,
    which should then be used for reweighting in the future.

    Parameters
    ----------
    freqs : np.ndarray
        frequency array at which to calculate Omega_GW
    mass_model : gwpopulation.models.mass._Mass
        mass model to use for the population model
    redshift_model : gwpopulation.models.redshift._Redshift
        redshift model to use for the population model
    initial_params : dict
        initial parameters to use for the population model
    num_proposal_samples : int, optional
        number of proposal samples to draw from the population model,
        to use later for reweighting by default 10000

    Returns
    -------
    pop_obj : popstock.PopulationOmegaGW.PopulationOmegaGW
        The population model object, initialized to the user-supplied initial parameters.
    """
    models = {'mass_model': mass_model, 'redshift_model': redshift_model}
    pop_obj = PopulationOmegaGW(frequency_array=freqs, models=models)
    pop_obj.draw_and_set_proposal_samples(initial_params,
                                          N_proposal_samples=num_proposal_samples)
    pop_obj.calculate_omega_gw(initial_params)
    return pop_obj

def create_initialize_popstock_plpp_md(freqs, params=plpp_md_default_params,
                                       num_proposal_samples=10000):
    """
    Create a population model using the PLPP mass distribution
    and the default redshift distribution.

    Parameters
    ----------
    freqs : np.ndarray
        frequency array at which to calculate Omega_GW
    params : dict, optional
        Default parameters to use for the population model,
        by default `plpp_md_default_params`.
        Note that these are the parameters that will be needed for
        reweighting in the future.
    num_proposal_samples : int, optional
        Number of proposal waveforms to draw from the population model,
        to use later for reweighting by default 10000

    Returns
    -------
    pop_obj : popstock.PopulationOmegaGW.PopulationOmegaGW
        The population model object.
    """
    mass_obj = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    redshift_obj = gwpopulation.models.redshift.MadauDickinsonRedshift(z_max=10)
    pop_obj = create_initialize_popstock_model(freqs, mass_obj, redshift_obj,
                                               params, num_proposal_samples)
    return pop_obj

class SplineRedshift(gwpopulation.models.redshift._Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
    See https://arxiv.org/abs/2003.12152 (2) for the normalisation

    The parameterisation differs a little from there, we use

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

    Parameters
    ----------
    gamma: float
        Slope of the distribution at low redshift
    kappa: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """
    base_variable_names = ["gamma", "kappa", "z_peak"]

    def psi_of_z(self, redshift, **parameters):
        amplitudes = np.array([parameters[key]
                              for key in parameters if 'amplitude' in key])
        configuration = np.array([parameters[key]
                                 for key in parameters if 'configuration' in key])
        xvals = np.array([parameters[key]
                         for key in parameters if 'xval' in key])
        return interp1d(xvals[configuration], amplitudes[configuration], fill_value="extrapolate")(redshift)


def createSplineRedshift(max_knots=10):
    class SplineRedshift(gwpopulation.models.redshift._Redshift):
        r"""
        Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
        See https://arxiv.org/abs/2003.12152 (2) for the normalisation

        The parameterisation differs a little from there, we use

        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) = interpolation with at most max_knots in z-space and a list of amplitudes

        Parameters
        ----------
        amplitudes: float
            Slope of the distribution at low redshift
        configuration: float
            Slope of the distribution at high redshift
        xvals: float
            Redshift at which the distribution peaks.
        """
        variable_names = [f"amplitudes{ii}" for ii in range(max_knots)] + \
            [f"configuration{ii}" for ii in range(max_knots)] + \
            [f"xvals{ii}" for ii in range(max_knots)]

        def psi_of_z(self, redshift, **parameters):
            amplitudes = np.array([parameters[key]
                                  for key in parameters if 'amplitude' in key])
            configuration = np.array(
                [parameters[key] for key in parameters if 'configuration' in key])
            xvals = np.array([parameters[key]
                             for key in parameters if 'xval' in key])
            if np.sum(configuration) == 0:
                tmp = 0 * np.zeros(redshift.size)
            elif np.sum(configuration) == 1:
                tmp = np.ones(redshift.size) * amplitudes[configuration]
            else:
                tmp = interp1d(xvals[configuration],
                               amplitudes[configuration],
                               fill_value="extrapolate")(redshift)
                # tmp = Akima1DInterpolator(xvals[configuration],
                #                           amplitudes[configuration])(redshift,
                #                                                 extrapolate=True)
            return tmp
    return SplineRedshift



def popstock_model(freqs, mass_obj, redshift_obj):
    """
    Create population OmegaGW model from a gwpopulations mass object and
    redshift object."""
    # choose models for calculating Omega_GW
    models = {'mass_model' : mass_obj,'redshift_model' : redshift_obj,}
    # Populations object requires mass and redshift distributions and frequencies
    pop_obj = PopulationOmegaGW(models=models, frequency_array=freqs)
    return pop_obj

def create_injected_OmegaGW(freqs, Lambda_0, N_proposal_samples, mass_obj, redshift_obj):
    """
    Create an injected Omega_GW
    """
    # create population object
    injection_pop = create_popOmegaGW(freqs, mass_obj, redshift_obj)

    # We need to define Λ_0 hyperparameters for population (must match formalism in the redshift and mass models) and number of desired samples
    injection_pop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)

    # Calculates Ω_GW
    injection_pop.calculate_omega_gw(Lambda=Lambda_0, multiprocess=True)

    # create a new set of Lambdas, with a different value of alpha
    injection_pop.calculate_omega_gw(Lambda=Lambda_0)

    # additional arguments needed to pass into sampling to use the spline redshift model
    args = np.argsort(injection_pop.proposal_samples['redshift'])

    # these can be our starting knots
    xvals = injection_pop.proposal_samples['redshift'][args][::100]
    # these can be our starting amplitudes
    amplitudes = injection_pop.models['redshift'].psi_of_z(injection_pop.proposal_samples['redshift'], **Lambda_0)[args][::100]
    # we'll start with all knots turned on
    configuration = np.ones(amplitudes.size, dtype=bool)

    # Keep the mass model the same, redshift model becomes the spline redshift
    splredshift = createSplineRedshift(amplitudes.size)(z_max=10)
    models = {'mass_model' : mass_obj,'redshift_model' : splredshift}

    # instantiate population object using the spline redshift model
    pop_for_sampling = PopulationOmegaGW(models=models, frequency_array=freqs, backend='numpy')

    # define the hyperparameters (need additional ones to Λ to match additional parameters needed for the spline redshift model)
    params_start = {**{f'amplitudes{ii}': amplitudes[ii] for ii in range(amplitudes.size)},
                    **{f'xvals{ii}': xvals[ii] for ii in range(xvals.size)},
                    **{f'configuration{ii}': configuration[ii] for ii in range(configuration.size)}}
    # now add our spline initial knots, amplitudes, configurations
    # to the parameters dictionary.
    Lambda_start = {**params_start, **Lambda_0}

    # number of waveforms to use for resampling
    N_proposal_samples = int(4.e4)

    # sample and calulate Ω_GW for spline redshift model
    # we're kinda cheating here by using a "fiducial" or "starting"
    # model that is the same as the "true" model.
    pop_for_sampling.draw_and_set_proposal_samples(Lambda_start, N_proposal_samples=N_proposal_samples)
    pop_for_sampling.calculate_omega_gw(Lambda=Lambda_start, multiprocess=True)

    return xvals, pop_for_sampling, injection_pop, amplitudes, configuration, Lambda_start