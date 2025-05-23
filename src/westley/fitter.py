import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from scipy.stats import norm
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from loguru import logger
import random

from typing import NamedTuple, Optional, Dict, Any, List
from dataclasses import dataclass

class ProposalResult(NamedTuple):
    log_likelihood: float
    log_ratio: float
    new_config: np.ndarray
    new_heights: np.ndarray
    new_knots: np.ndarray

@dataclass(frozen=True)
class SplineState:
    configuration: np.ndarray
    heights: np.ndarray
    knots: np.ndarray
    log_likelihood: Optional[float] = None

    def with_updates(self, **kwargs):
        return SplineState(**{**self.__dict__, **kwargs})

class ProposalManager:
    def __init__(self):
        self._proposals = {}
        self._weights = {}
    
    def register_proposal(self, name, func, weight=1):
        self._proposals[name] = func
        self._weights[name] = weight
    
    def get_next_proposal(self):
        proposals = list(self._proposals.keys())
        weights = list(self._weights.values())
        return random.choices(proposals, weights=np.array(weights)/sum(weights), k=1)[0]
    
    @property
    def proposals(self):
        return self._proposals
    
    @proposals.setter
    def proposals(self, new_proposals):
        old_proposals = dict(self._proposals)
        self._proposals = {}
        self._weights = {}
        for name, weight in new_proposals.items():
            if name in old_proposals:
                self._proposals[name] = old_proposals[name]
                self._weights[name] = weight
            else:
                raise ValueError(f"Proposal {name} is not registered.")
        
    @property
    def weights(self):
        return self._weights

def proposal(name, weight=1):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        wrapper._proposal_name = name
        wrapper._proposal_weight = weight
        return wrapper
    return decorator

class SamplerState(NamedTuple):
    state: SplineState
    proposal_type: str
    accepted: bool
    log_ratio: float

class BaseSplineModel:
    def __init__(self, data, N_possible_knots, xrange, height_prior_range, 
                 min_knots=2, birth_uniform_frac=0.5, birth_gauss_scalefac=0.5,
                 log_output=False, log_space_xvals=False, interp_type="linear",
                 temperature=1.0, proposal_cycle=None):
        
        self.data = data
        self.N_possible_knots = N_possible_knots
        self.min_knots = min_knots
        self.xlow = xrange[0] + 1e-3
        self.xhigh = xrange[1]
        self.yhigh = height_prior_range[1]
        self.ylow = height_prior_range[0]
        self.yrange = self.yhigh - self.ylow
        self.birth_uniform_frac = birth_uniform_frac
        self.birth_gauss_scalefac = birth_gauss_scalefac
        self.interp_type = interp_type
        self.temperature = temperature  # Store temperature as an attribute
        self.proposal_cycle = proposal_cycle  # Store proposal_cycle as an attribute

        self.init_args = (data, N_possible_knots, xrange, height_prior_range)
        self.init_kwargs = {
            "min_knots": min_knots,
            "birth_uniform_frac": birth_uniform_frac,
            "birth_gauss_scalefac": birth_gauss_scalefac,
            "log_output": log_output,
            "log_space_xvals": log_space_xvals,
            "interp_type": interp_type,
            "temperature": temperature,
            "proposal_cycle": proposal_cycle,
        }

        if log_space_xvals:
            base = np.logspace(np.log10(self.xlow), np.log10(self.xhigh), 
                             num=self.N_possible_knots + 1)
            self.xlows = base[:-1]
            self.xhighs = base[1:]
            self.available_knots = (self.xlows + self.xhighs) / 2
        else:
            self.deltax = (self.xhigh - self.xlow) / N_possible_knots
            self.xlows = np.arange(N_possible_knots) * self.deltax + self.xlow
            self.xhighs = self.xlows + self.deltax
            self.available_knots = np.linspace(self.xlow + self.deltax / 2, 
                                             self.xhigh - self.deltax / 2, 
                                             num=self.N_possible_knots)

        initial_log_likelihood = self.ln_likelihood(
            np.ones(self.N_possible_knots, dtype=bool),
            np.ones(self.N_possible_knots) * (self.yhigh - self.ylow) / 2. + self.ylow,
            self.available_knots.copy()
        )
        self.state = SplineState(
            configuration=np.ones(self.N_possible_knots, dtype=bool),
            heights=np.ones(self.N_possible_knots) * (self.yhigh - self.ylow) / 2. + self.ylow,
            knots=self.available_knots.copy(),
            log_likelihood=initial_log_likelihood
        )
        
        self.proposal_manager = ProposalManager()
        self.log_output = log_output
        self.sampler_state = None  # Initialize sampler_state attribute

        # Register proposals
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_proposal_name'):
                self.proposal_manager.register_proposal(attr._proposal_name, attr, attr._proposal_weight)

        # Backward compatibility
        self.configuration = self.state.configuration
        self.current_heights = self.state.heights
        self._proposal_cycle = self.proposal_manager.weights
        self._available_proposals = list(self.proposal_manager.proposals.keys())

        if self.proposal_cycle is not None:
            self.proposal_manager.proposals = self.proposal_cycle

    def copy(self):
        """
        Create a new instance of the model with the same initialization arguments.
        """
        return type(self)(*self.init_args, **self.init_kwargs)

    def set_proposal_cycle(self, proposal_cycle: Dict[str, float]):
        """
        Update the proposal cycle and ensure consistency with the ProposalManager.
        """
        self.proposal_cycle = proposal_cycle
        self.proposal_manager.proposals = self.proposal_cycle

    def ln_likelihood(self, config, heights, knots):
        """
        Abstract method for log-likelihood calculation.
        Subclasses must override this method to define their specific likelihood.
        """
        raise NotImplementedError("Subclasses must implement ln_likelihood")

    def _calculate_birth_ratio(self, idx_to_add, new_heights, new_knots):
        """Calculate the log ratio for a birth proposal."""
        height_from_model = self.evaluate_interp_model(
            new_knots[idx_to_add],
            self.state.heights,
            self.state.configuration,
            self.state.knots
        )
        
        log_qx = 0
        log_qy = np.log(
            self.birth_uniform_frac / self.yrange + 
            (1 - self.birth_uniform_frac) * norm.pdf(
                new_heights[idx_to_add],
                loc=height_from_model,
                scale=self.birth_gauss_scalefac
            )
        )
        
        log_px = 0
        log_py = self.get_height_log_prior(new_heights[idx_to_add])
        
        proposal_weights = self.proposal_manager.weights
        weight_birth = proposal_weights.get('birth', 1.0)
        weight_death = proposal_weights.get('death', 1.0)
        log_ratio = (log_py - log_px) + (log_qx - log_qy) + np.log(weight_death / weight_birth)
        return log_ratio

    def _calculate_death_ratio(self, idx_to_remove, heights, knots):
        """Calculate the log ratio for a death proposal."""
        new_config = self.state.configuration.copy()
        new_config[idx_to_remove] = False
        height_from_model = self.evaluate_interp_model(
            knots[idx_to_remove],
            heights,
            new_config,
            knots
        )

        log_qy = 0
        log_qx = np.log(
            self.birth_uniform_frac / self.yrange + 
            (1 - self.birth_uniform_frac) * norm.pdf(
                heights[idx_to_remove],
                loc=height_from_model,
                scale=self.birth_gauss_scalefac
            )
        )

        log_py = 0
        log_px = self.get_height_log_prior(heights[idx_to_remove])

        proposal_weights = self.proposal_manager.weights
        weight_birth = proposal_weights.get('birth', 1.0)
        weight_death = proposal_weights.get('death', 1.0)
        log_ratio = (log_py - log_px) + (log_qx - log_qy) + np.log(weight_birth / weight_death)
        return log_ratio

    def get_height_log_prior(self, height):
        """Calculate the log prior for a given height."""
        if height < self.ylow or height > self.yhigh:
            return -np.inf
        return -np.log(self.yrange)

    @proposal(name='birth', weight=1)
    def birth(self):
        """Birth proposal: Add a new knot."""
        inactive_idx = np.where(~self.state.configuration)[0]
        if len(inactive_idx) == 0:
            return None
            
        idx_to_add = random.choices(inactive_idx, k=1)[0]
        new_config = self.state.configuration.copy()
        new_config[idx_to_add] = True
        
        new_knots = self.state.knots.copy()
        new_heights = self.state.heights.copy()
        
        new_knots[idx_to_add] = (np.random.rand() * 
            (self.xhighs[idx_to_add] - self.xlows[idx_to_add]) + 
            self.xlows[idx_to_add])
        
        height_from_model = self.evaluate_interp_model(
            new_knots[idx_to_add],
            self.state.heights,
            self.state.configuration,
            self.state.knots
        )
        
        if np.random.rand() < self.birth_uniform_frac:
            new_heights[idx_to_add] = (np.random.rand() * 
                (self.yhigh - self.ylow) + self.ylow)
        else:
            new_heights[idx_to_add] = norm.rvs(
                loc=height_from_model,
                scale=self.birth_gauss_scalefac
            )
        
        log_ratio = self._calculate_birth_ratio(idx_to_add, new_heights, new_knots)
        new_ll = self.ln_likelihood(new_config, new_heights, new_knots)
        
        return ProposalResult(new_ll, log_ratio, new_config, new_heights, new_knots)

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

    @proposal(name='change_amplitude_prior_draw', weight=1)
    def change_amplitude_prior_draw(self):
        """Change amplitude by drawing from the prior."""
        if self.state.configuration.sum() == 0:
            return None
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = random.choices(active_idx, k=1)[0]

        new_heights = self.state.heights.copy()
        new_heights[idx_to_change] = (np.random.rand() * 
            (self.yhigh - self.ylow) + self.ylow)

        log_py_before = self.get_height_log_prior(self.state.heights[idx_to_change])
        log_py_after = self.get_height_log_prior(new_heights[idx_to_change])
        log_ratio = log_py_after - log_py_before

        new_ll = self.ln_likelihood(
            self.state.configuration, new_heights, self.state.knots)

        return ProposalResult(
            new_ll, log_ratio, self.state.configuration.copy(),
            new_heights, self.state.knots.copy()
        )

    @proposal(name='change_amplitude_gaussian', weight=1)
    def change_amplitude_gaussian(self):
        """Change amplitude using a Gaussian proposal."""
        if self.state.configuration.sum() == 0:
            return None
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = random.choices(active_idx, k=1)[0]

        new_heights = self.state.heights.copy()
        new_heights[idx_to_change] += norm.rvs(scale=self.birth_gauss_scalefac)

        log_py_before = self.get_height_log_prior(self.state.heights[idx_to_change])
        log_py_after = self.get_height_log_prior(new_heights[idx_to_change])
        log_ratio = log_py_after - log_py_before

        new_ll = self.ln_likelihood(
            self.state.configuration, new_heights, self.state.knots)

        return ProposalResult(
            new_ll, log_ratio, self.state.configuration.copy(),
            new_heights, self.state.knots.copy()
        )

    @proposal(name='change_knot_location', weight=1)
    def change_knot_location(self):
        """Change the location of an existing knot."""
        if self.state.configuration.sum() == 0:
            return None
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = random.choices(active_idx, k=1)[0]
        
        new_knots = self.state.knots.copy()
        new_knots[idx_to_change] = (np.random.rand() * 
            (self.xhighs[idx_to_change] - self.xlows[idx_to_change]) + 
            self.xlows[idx_to_change])
        
        new_ll = self.ln_likelihood(
            self.state.configuration, self.state.heights, new_knots)
        
        return ProposalResult(
            new_ll, 0.0, self.state.configuration.copy(),
            self.state.heights.copy(), new_knots
        )

    def evaluate_interp_model(self, x, heights, configuration, knots):
        active_knots = knots[configuration]
        active_heights = heights[configuration]

        if len(active_knots) < 2:
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

    def step(self, prior_test=False) -> SamplerState:
        """Make a single MCMC step and update the sampler state."""
        # Update proposal manager if a proposal cycle is set
        if self.proposal_cycle is not None:
            self.proposal_manager.proposals = self.proposal_cycle
        
        # Get proposal type
        proposal_type = self.proposal_manager.get_next_proposal()
        
        # Get proposal
        proposal = self.proposal_manager.proposals[proposal_type]()
        
        if proposal is None:
            # No valid proposal, return current state
            self.sampler_state = SamplerState(
                state=self.state,
                proposal_type=proposal_type,
                accepted=False,
                log_ratio=0.0
            )
            return self.sampler_state
        
        # Calculate acceptance probability
        if prior_test:
            log_alpha = proposal.log_ratio
        else:
            if self.state.log_likelihood is None:
                raise ValueError("log_likelihood in the current state is not initialized.")
            log_alpha = (proposal.log_likelihood - self.state.log_likelihood) / self.temperature + proposal.log_ratio
        
        # Accept/reject
        accepted = np.log(np.random.rand()) < log_alpha
        if accepted:
            # Directly update self.state if accepted
            self.state = SplineState(
                configuration=proposal.new_config,
                heights=proposal.new_heights,
                knots=proposal.new_knots,
                log_likelihood=proposal.log_likelihood
            )
        
        # Update sampler_state to reflect the current state
        self.sampler_state = SamplerState(
            state=self.state,
            proposal_type=proposal_type,
            accepted=accepted,
            log_ratio=proposal.log_ratio
        )
        return self.sampler_state

    def run(self, Niterations=1000, prior_test=False, start_heights=None, start_knots=None):
        """
        Run MCMC sampling for specified number of iterations.

        :param Niterations: Number of iterations to run the sampler.
        :param prior_test: If True, only test the prior without likelihood.
        :param start_heights: Initial heights for the knots.
        :param start_knots: Initial knot locations.
        """
        # Validate and set initial state
        if start_heights is not None and start_knots is not None:
            if len(start_heights) != self.N_possible_knots or len(start_knots) != self.N_possible_knots:
                raise ValueError("start_heights and start_knots must have the same length as N_possible_knots.")
            if not np.all((self.ylow <= start_heights) & (start_heights <= self.yhigh)):
                raise ValueError("start_heights must be within the prior bounds.")
            if not np.all((self.xlow <= start_knots) & (start_knots <= self.xhigh)):
                raise ValueError("start_knots must be within the prior bounds.")
            self.state = SplineState(
                configuration=self.state.configuration,
                heights=start_heights,
                knots=start_knots,
                log_likelihood=self.ln_likelihood(
                    self.state.configuration, start_heights, start_knots
                )
            )
        elif start_heights is not None or start_knots is not None:
            raise ValueError("If you provide a starting state, you must provide both start_heights and start_knots.")

        # Ensure initial log-likelihood is valid
        if self.state.log_likelihood is None:
            self.state = self.state.with_updates(
                log_likelihood=self.ln_likelihood(
                    self.state.configuration, self.state.heights, self.state.knots
                )
            )

        logger.info(f"Proposal cycle: {self.proposal_manager.proposals}")
        
        # Initialize arrays to store chain
        configurations = np.zeros((Niterations, self.N_possible_knots), dtype=bool)
        heights = np.zeros((Niterations, self.N_possible_knots))
        knots = np.zeros((Niterations, self.N_possible_knots))
        log_likelihoods = np.zeros(Niterations)
        proposals_used = []
        acceptances = np.zeros(Niterations, dtype=bool)
        log_ratios = np.zeros(Niterations)
        
        # Store initial state
        configurations[0] = self.state.configuration
        heights[0] = self.state.heights
        knots[0] = self.state.knots
        log_likelihoods[0] = (self.ln_likelihood(
            self.state.configuration,
            self.state.heights,
            self.state.knots
        ) if not prior_test else 0)
        
        n_accepted = 0
        
        with tqdm(total=Niterations, desc="Sampling", unit="iter") as pbar:
            for i in range(1, Niterations):
                # Make one MCMC step
                self.step(prior_test=prior_test)
                self.state = self.sampler_state.state
                
                # Store results
                configurations[i] = self.state.configuration
                heights[i] = self.state.heights
                knots[i] = self.state.knots
                log_likelihoods[i] = self.state.log_likelihood
                proposals_used.append(self.sampler_state.proposal_type)
                acceptances[i] = self.sampler_state.accepted
                log_ratios[i] = self.sampler_state.log_ratio
                
                if self.sampler_state.accepted:
                    n_accepted += 1
                    
                # Update progress bar
                pbar.set_postfix(acceptance_rate=n_accepted/(i+1))
                pbar.update(1)

        if self.log_output:
            print(f"Acceptance rate: {n_accepted/(Niterations-1):.3f}")

        return SamplerResults(
            log_likelihoods=log_likelihoods,
            configurations=configurations,
            heights=heights,
            knots=knots,
            acceptance_rate=n_accepted/(Niterations),
            proposals_used=proposals_used,
            acceptances=acceptances,
            log_ratios=log_ratios
        )

class SamplerResults:
    """
    Class to handle results from the sampling process.
    """
    def __init__(self, log_likelihoods, configurations, heights, knots,
                 acceptance_rate, proposals_used,
                 acceptances, log_ratios, swap_acceptance_rate=None):
        self.log_likelihoods = log_likelihoods
        self.configurations = configurations
        self.heights = heights
        self.knots = knots
        self.acceptance_rate = acceptance_rate
        self.proposals_used = proposals_used
        self.acceptances = acceptances
        self.log_ratios = log_ratios
        self.swap_acceptance_rate = swap_acceptance_rate

    def get_num_knots(self):
        """Return the number of active knots for each sample."""
        return np.sum(self.configurations, axis=1)

    def return_knot_info(self, offset=0):
        """Return knot configurations and counts."""
        knot_configs = self.configurations
        num_knots = knot_configs.sum(axis=1)
        return knot_configs, num_knots

    def return_knot_placements(self, offset=0):
        """Return histogram data for knot placements."""
        all_weights = []
        all_bins = []
        for ii in range(self.knots.shape[1]):
            if np.sum(self.configurations.astype(bool)[offset:, ii]) > 0:    
                weights, bins, _ = plt.hist(self.knots[self.configurations.astype(bool)[:, ii], ii])
                all_weights.append(weights)
                all_bins.append(bins[:-1])
        return all_bins, all_weights

    def return_knot_heights(self, offset=0, toggle=False):
        """Return heights of active knots."""
        if toggle:
            knot_heights = []
            for ii in range(self.knots.shape[1]):
                knot_heights.append(10**(self.heights[self.configurations.astype(bool)[:, ii], ii]))
        else: 
            knot_heights = 10**self.heights[offset:, :]
        return knot_heights

    def return_knot_frequencies(self, offset=0, toggle=False):
        """Return frequencies of active knots."""
        temp = []
        for ii in range(self.knots.shape[1]):
            if toggle:
                temp.append(self.knots[self.configurations.astype(bool)[:, ii], ii])
            else:
                temp.append(self.knots[:, ii])
        return temp

    def print_acceptance_rates_by_proposal(self):
        """Print acceptance rates for each proposal."""
        from collections import Counter
        total_counts = Counter(self.proposals_used)
        accepted_counts = Counter()
        
        for i, proposal in enumerate(self.proposals_used):
            if self.acceptances[i]:
                accepted_counts[proposal] += 1
        
        for proposal, count in total_counts.items():
            rate = accepted_counts[proposal] / count if count else 0
            print(f"{proposal}: {rate:.3f} {accepted_counts[proposal]}")

class SmoothCurveDataObj:
    """
    A data class that can be used with our spline model.
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitSmoothCurveModel(BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `SmoothCurveDataObj`.
    """
    def ln_likelihood(self, config, heights, knots):
        """
        Simple Gaussian log-likelihood where the data are points in 2D space
        that we're trying to fit.

        :param config: Active configuration of knots.
        :param heights: Heights of the knots.
        :param knots: Knot locations.
        :return: Log-likelihood value.
        """
        model = self.evaluate_interp_model(self.data.data_xvals, heights, config, knots)
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))