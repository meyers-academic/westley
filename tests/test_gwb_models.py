from westley.gwb_models import FitOmega, RedshiftSampler
from westley.simulate import (get_sigma_from_noise_curves,
                              simulate_broken_powerlaw)
import numpy as np
import matplotlib.pyplot as plt

from westley.fitter import SmoothCurveDataObj

from westley.population_models import (create_initialize_popstock_model, 
                                        create_initialize_popstock_plpp_md)

from westley.population_models import SplineRedshift, createSplineRedshift
from gwpopulation.models import mass
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def test_fit_omega():
    """Just a test to make sure FitOmega class runs,
    and that the acceptance rate is not zero. No guarantees
    that the results are correct.
    """
    freqs = np.arange(20, 1000, 0.25)
    Tobs = 86400 * 365.25
    detectors = ['H1', 'L1']
    sigma = get_sigma_from_noise_curves(detectors, freqs, Tobs)
    signal, data, data_obj = simulate_broken_powerlaw(freqs, sigma, 1e-8)
    fit_omega = FitOmega(data_obj, 10, (10, 1000), (-15, -3))
    results = fit_omega.sample(10000)

    assert results is not None
    assert np.sum(results.acceptances) > 0, f"Acceptance rate should not be zero"

def test_spline_redshift_run():
    """Just a test to make sure SplineRedshift class runs,
    and that the acceptance rate is not zero. No guarantees
    that the results are correct.
    """
    initial_params = {
        'alpha': 2.5, 
        'beta': 1, 
        'delta_m': 3, 
        'lam': 0.04, 
        'mmax': 100,
        'mmin': 4, 
        'mpp': 33, 
        'sigpp': 5,
        'gamma': 2.7, 
        'kappa': 3, 
        'z_peak': 1.9, 
        'rate': 15
    }
    
    max_knots = 20
    freqs = np.arange(20, 100, 0.25)
    Tobs = 86400 * 365.25 * 1000
    detectors = ['H1', 'L1']
    num_samples =100

    redshift_knot_vals = np.linspace(0, 10, max_knots)
    # for injection
    inj_pop_obj = create_initialize_popstock_plpp_md(freqs=freqs,
                                                     num_proposal_samples=10000)

    # sorted red shift arguments
    args = np.argsort(inj_pop_obj.proposal_samples['redshift'])
    # get R(z) at the sorted redshifts
    redshift_heights = inj_pop_obj.models['redshift'].psi_of_z(inj_pop_obj.proposal_samples['redshift'], **initial_params)[args]
    redshift_values = inj_pop_obj.proposal_samples['redshift'][args]

    # interpolate R(z) at the knots
    r_of_z_at_knots = np.interp(redshift_knot_vals, redshift_values, redshift_heights)

    spline_redshift = createSplineRedshift(max_knots=max_knots)(z_max=max(redshift_knot_vals))

    mass_model = mass.SinglePeakSmoothedMassDistribution()
    params_start = {**{f'amplitudes{ii}': r_of_z_at_knots[ii] for ii in range(r_of_z_at_knots.size)}, 
                    **{f'xvals{ii}': redshift_knot_vals[ii] for ii in range(redshift_knot_vals.size)}, 
                    **{f'configuration{ii}': True for ii in range(r_of_z_at_knots.size)}}
    params_start = {**params_start, **initial_params}
    models = {'mass_model' : mass_model,'redshift_model' : spline_redshift}

    # instantiate population object using the spline redshift model 
    pop = create_initialize_popstock_model(freqs=freqs,
                                          mass_model=mass_model,
                                          redshift_model=spline_redshift,
                                          initial_params=params_start,
                                          num_proposal_samples=100)

    pop.calculate_omega_gw(params_start)

    # create fake data with some noise
    sigma = get_sigma_from_noise_curves(detectors, freqs, Tobs) 

    data = inj_pop_obj.omega_gw + np.random.normal(0, sigma, size=inj_pop_obj.omega_gw.size)
    data_obj = SmoothCurveDataObj(freqs, data, sigma)
    sampler = RedshiftSampler(data_obj, max_knots, (0, 10), (0, 100))
    sampler.set_base_population_information(pop, params_start)

    results = sampler.sample(num_samples, start_heights=r_of_z_at_knots,
                             start_knots=redshift_knot_vals)

    assert results is not None
    assert np.sum(results.acceptances) > 0, "Acceptance rate should not be zero"
