from typing import List, Optional, Dict
import numpy as np
from copy import deepcopy
from .fitter import BaseSplineModel, SamplerResults
from tqdm import tqdm
from loguru import logger

class ParallelTemperingResults:
    def __init__(self, chain_results: List[List[SamplerResults]], swap_rates: Dict[int, float], temperature_history: List[np.ndarray]):
        """
        Encapsulates the results of parallel tempering sampling.

        :param chain_results: A list of lists containing the sampler results for each chain.
        :param swap_rates: A dictionary where keys are chain indices and values are swap rates.
        :param temperature_history: A list of temperature arrays for each step.
        """
        self.chain_results = chain_results
        self.swap_rates = swap_rates
        self.temperature_history = temperature_history

    def get_chain_results(self, chain_idx: int) -> List[SamplerResults]:
        """Get the results for a specific chain."""
        return self.chain_results[chain_idx]

    def get_swap_rate(self, chain_idx: int) -> float:
        """Get the swap rate for a specific chain."""
        return self.swap_rates.get(chain_idx, 0.0)

    def get_temperature_history(self) -> List[np.ndarray]:
        """Get the temperature history for all chains."""
        return self.temperature_history

class ParallelTempering:
    def __init__(self, base_model: BaseSplineModel, n_temps: int = 10, max_temp: float = 50.0, adapt_ladder: bool = False, t_0: int = 1000, nu: int = 10):
        """
        Initialize the Parallel Tempering sampler.

        :param base_model: The base model to be sampled.
        :param n_temps: Number of temperature chains.
        :param max_temp: Maximum temperature.
        :param adapt_ladder: Whether to adaptively adjust the temperature ladder.
        :param t_0: Parameter controlling the decay of the adaptation rate.
        """
        self.n_temps = n_temps
        self.max_temp = max_temp
        self.adapt_ladder = adapt_ladder
        self.t_0 = t_0  # Controls how quickly the adaptation rate decays
        self.temperatures = np.exp(np.linspace(0, np.log(max_temp), n_temps))
        self.target_swap_rate = 0.234  # Target swap acceptance rate (Vousden et al., 2016)
        self.swap_rate_tolerance = 0.05  # Tolerance for swap rate adjustment
        self.nu = nu

        # Create chain copies at different temperatures
        self.chains = [base_model]
        for _ in range(n_temps - 1):
            self.chains.append(base_model.copy())  # Use the copy method instead of deepcopy

    def _adaptation_rate(self, step: int) -> float:
        """
        Compute the adaptation rate as a function of the current step.

        :param step: The current sampling step.
        :return: The adaptation rate, which decreases as step increases.
        """
        return 1 / (self.nu * (1 + step / self.t_0))

    def _adjust_temperatures(self, swap_stats: np.ndarray, step: int):
        """
        Adjust the temperature ladder based on observed swap rates.

        :param swap_stats: Array of swap attempts and acceptances.
        :param step: The current sampling step.
        """
        swap_rates = swap_stats[:, 1] / np.maximum(swap_stats[:, 0], 1)  # Avoid division by zero
        # logger.info(f"Observed swap rates: {swap_rates}")

        adaptation_rate = self._adaptation_rate(step)
        # logger.info(f"Adaptation rate at step {step}: {adaptation_rate:.5f}")
        diffs = np.diff(self.temperatures)

        for i in range(1, len(swap_rates)):  # Start from index 1 to skip the cold chain
            # print('diffs', np.log(diffs[i-1]))
            # print('arg_of_exp', adaptation_rate*(swap_rates[i]-swap_rates[i-1]))
            self.temperatures[i] = self.temperatures[i-1] + np.exp(np.log(diffs[i-1])+adaptation_rate*(-swap_rates[i]+swap_rates[i-1]))


        # Ensure temperatures remain sorted and the cold chain temperature stays fixed at T=1
        self.temperatures[1:] = np.sort(self.temperatures[1:])
        self.temperatures[0] = 1.0  # Fix the cold chain temperature

        # Update the temperature of each chain
        for chain, temp in zip(self.chains, self.temperatures):
            chain.temperature = temp

    def _swap_attempt(self, i: int) -> bool:
        """Attempt temperature swap between chains i and i+1"""
        T1, T2 = self.temperatures[i], self.temperatures[i+1]
        L1 = self.chains[i].state.log_likelihood
        L2 = self.chains[i+1].state.log_likelihood

        # Compute the prior probabilities

        # Compute the log acceptance probability
        log_alpha = (1/T1 - 1/T2) * (L2 - L1)

        if np.log(np.random.rand()) < log_alpha:
            # Swap SplineState (not SamplerState) between chains
            self.chains[i].state, self.chains[i+1].state = self.chains[i+1].state, self.chains[i].state
            return True
        return False

    def sample(self, Niterations: int, swap_interval: int = 10, proposal_cycle=None, **kwargs) -> ParallelTemperingResults:
        """Sample all chains with periodic temperature swaps."""
        results = [[] for _ in range(self.n_temps)]
        swap_stats = np.zeros((self.n_temps - 1, 2))  # attempts, accepts
        temperature_history = []  # Track temperature history

        logger.info("Initializing chains")
        # Initialize all chains
        for chain, temp in zip(self.chains, self.temperatures):
            chain.temperature = temp  # Set temperature directly
            chain.proposal_cycle = proposal_cycle  # Set proposal_cycle directly
            chain.step(**kwargs)  # Perform a single step to initialize sampler_state

        # Main sampling loop
        for i in tqdm(range(Niterations)):
            # Sample each chain
            for kk, chain in enumerate(self.chains):
                chain.step(**kwargs)  # Perform a single step
                results[kk].append(chain.sampler_state)  # Store sampler_state

            # Record the current temperatures
            temperature_history.append(self.temperatures.copy())

            # Skip temperature swaps if only one chain
            if self.n_temps == 1:
                continue

            # Try temperature swaps
            if i % swap_interval == 0:
                for j in reversed(range(self.n_temps - 1)):
                    swap_stats[j, 0] += 1  # Increment attempts
                    if self._swap_attempt(j):
                        swap_stats[j, 1] += 1  # Increment accepts

                # Adjust temperature ladder if adaptation is enabled
                if self.adapt_ladder:
                    self._adjust_temperatures(swap_stats, i)

        # Calculate swap rates
        swap_rates = {j: swap_stats[j, 1] / swap_stats[j, 0] if swap_stats[j, 0] > 0 else 0.0 for j in range(self.n_temps - 1)}
        for j, rate in swap_rates.items():
            logger.info(f"Swap rate between chains {j} and {j+1}: {rate:.3f}")

        return ParallelTemperingResults(results, swap_rates, temperature_history)
