import numpy as np
from westley.fitter import FitSmoothCurveModel, SmoothCurveDataObj
from westley.parallel_tempering import ParallelTempering
import matplotlib.pyplot as plt

def test_parallel_tempering_birth_death():
    """Test parallel tempering with simple birth-death process"""
    # Create simple test data
    xvals = np.linspace(0, 10, 20)
    true_y = np.sin(xvals)
    errors = np.ones_like(xvals) * 0.8
    data = true_y + np.random.normal(0, errors)
    
    # Create base model
    data_obj = SmoothCurveDataObj(xvals, data, errors)
    base_model = FitSmoothCurveModel(
        data_obj, 10, (0, 10), (-2, 2),
        min_knots=2, birth_uniform_frac=0.5
    )
    
    # Setup parallel tempering
    n_temps = 3
    max_temp = 10
    pt = ParallelTempering(base_model, n_temps=n_temps, max_temp=max_temp)
    
    # Sample with parallel tempering
    pt_results = pt.sample(
        Niterations=10000,
        swap_interval=5,
    )

    # Sample with the base model
    results_base = base_model.run(Niterations=100000, prior_test=False)

    # Basic checks
    assert len(pt_results.chain_results) == n_temps, "Number of results should match number of temperatures"
    assert results_base is not None, "Base model sampling results should not be None"

    # Retrieve and print swap rates
    swap_rates = pt_results.swap_rates
    print(f"Swap rates: {swap_rates}")
    assert all(0.1 < rate < 0.9 for rate in swap_rates.values()), "Swap rates should be moderate"

    # Plot recovered fits for a few draws from the cold chain (parallel tempering)
    cold_chain = pt_results.get_chain_results(0)  # Cold chain is the first chain
    plt.figure(figsize=(10, 6))
    idxs = np.random.randint(5000, len(cold_chain), 500)  # Randomly select 500 samples
    for i in idxs:
        state = cold_chain[i].state
        yvals = base_model.evaluate_interp_model(data_obj.data_xvals, state.heights, state.configuration, state.knots)
        plt.plot(data_obj.data_xvals, yvals, alpha=0.1, c='r')
    plt.errorbar(data_obj.data_xvals, data, yerr=errors, fmt='o', label="Data", color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Recovered Fits from Cold Chain (Parallel Tempering)")
    plt.legend()
    plt.savefig("recovered_fits_cold_chain_pt.png")
    plt.close()

    # Plot recovered fits for a few draws from the base model
    plt.figure(figsize=(10, 6))
    idxs = np.random.randint(5000, len(results_base.configurations), 500)  # Randomly select 500 samples
    for i in idxs:
        yvals = base_model.evaluate_interp_model(
            data_obj.data_xvals,
            results_base.heights[i],
            results_base.configurations[i],
            results_base.knots[i]
        )
        plt.plot(data_obj.data_xvals, yvals, alpha=0.1, c='b')
    plt.errorbar(data_obj.data_xvals, data, yerr=errors, fmt='o', label="Data", color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Recovered Fits from Base Model")
    plt.legend()
    plt.savefig("recovered_fits_base_model.png")
    plt.close()

    # Plot total number of knots as a function of step for all chains (parallel tempering)
    plt.figure(figsize=(10, 6))
    for chain_idx, chain in enumerate(pt_results.chain_results):
        num_knots = [np.sum(state.state.configuration) for state in chain]
        plt.plot(num_knots, label=f"Chain {chain_idx}")
    plt.xlabel("Step")
    plt.ylabel("Number of Knots")
    plt.title("Number of Knots vs Step (Parallel Tempering)")
    plt.legend()
    plt.savefig("num_knots_vs_step_pt.png")
    plt.close()

    # Plot total number of knots as a function of step for the base model
    plt.figure(figsize=(10, 6))
    num_knots_base = results_base.get_num_knots()
    plt.plot(num_knots_base, label="Base Model")
    plt.xlabel("Step")
    plt.ylabel("Number of Knots")
    plt.title("Number of Knots vs Step (Base Model)")
    plt.legend()
    plt.savefig("num_knots_vs_step_base_model.png")
    plt.close()

def test_single_chain_behavior():
    """Test parallel tempering with a single chain (should behave like normal sampling)"""
    # Create simple test data
    xvals = np.linspace(0, 10, 20)
    true_y = np.sin(xvals)
    errors = np.ones_like(xvals) * 0.1
    data = true_y + np.random.normal(0, errors)
    
    # Create base model
    data_obj = SmoothCurveDataObj(xvals, data, errors)
    base_model = FitSmoothCurveModel(
        data_obj, 10, (0, 10), (-2, 2),
        min_knots=2, birth_uniform_frac=0.5
    )
    
    # Setup parallel tempering with a single chain
    pt = ParallelTempering(base_model, n_temps=1, max_temp=1.0)
    
    # Sample
    results = pt.sample(Niterations=1000, swap_interval=5, prior_test=True)
    
    # Check results
    assert len(results.chain_results) == 1, "There should be only one chain"
    assert len(results.chain_results[0]) == 1000, "Number of iterations should match"
    num_knots = [np.sum(state.state.configuration) for state in results.get_chain_results(0)]
    print('KNOTS', num_knots)
    assert len(set(num_knots)) > 1, "Sampler is stuck in a single configuration"
