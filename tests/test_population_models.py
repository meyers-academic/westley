import numpy as np
import pytest
import gwpopulation
from westley.population_models import (create_initialize_popstock_model, 
                                        create_initialize_popstock_plpp_md)


def test_create_initialize_popstock_model_runs():
    # Setup test inputs
    freqs = np.logspace(0, 3, 100)
    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    redshift_model = gwpopulation.models.redshift.MadauDickinsonRedshift(z_max=10)
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

    # Run function
    pop_obj = create_initialize_popstock_model(
        freqs=freqs,
        mass_model=mass_model,
        redshift_model=redshift_model,
        initial_params=initial_params,
        num_proposal_samples=100  # Using a small number for quick testing
    )

    # Basic assertions to ensure the object was created and initialized
    assert pop_obj is not None
    assert hasattr(pop_obj, 'frequency_array')
    assert hasattr(pop_obj, 'models')
    assert hasattr(pop_obj, 'proposal_samples')



def test_create_initialize_popstock_plpp_md_runs():
    # Setup test inputs
    freqs = np.logspace(0, 3, 100)

    # Run function with default parameters
    pop_obj = create_initialize_popstock_plpp_md(
        freqs=freqs,
        num_proposal_samples=100  # Using a small number for quick testing
    )

    # Basic assertions to ensure the object was created and initialized
    assert pop_obj is not None
    assert hasattr(pop_obj, 'frequency_array')
    assert hasattr(pop_obj, 'models')
    assert hasattr(pop_obj, 'proposal_samples')

    # Verify the correct models were used
    assert isinstance(pop_obj.models['mass'], 
                      gwpopulation.models.mass.SinglePeakSmoothedMassDistribution)
    assert isinstance(pop_obj.models['redshift'], 
                      gwpopulation.models.redshift.MadauDickinsonRedshift)