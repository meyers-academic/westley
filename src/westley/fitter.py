import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from scipy.stats import norm
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from loguru import logger



from typing import NamedTuple, Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from copy import deepcopy
from scipy.interpolate import interp1d, Akima1DInterpolator

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
        return np.random.choice(proposals, p=np.array(weights)/sum(weights))
    
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

class BaseSplineModel:
    def __init__(self, data, N_possible_knots, xrange, height_prior_range, 
                 min_knots=2, birth_uniform_frac=0.5, birth_gauss_scalefac=0.5,
                 log_output=False, log_space_xvals=False, interp_type="linear"):
        
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

        self.state = SplineState(
            configuration=np.ones(self.N_possible_knots, dtype=bool),
            heights=np.ones(self.N_possible_knots) * (self.yhigh - self.ylow) / 2. + self.ylow,
            knots=self.available_knots.copy()
        )
        
        self.proposal_manager = ProposalManager()
        self.log_output = log_output

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
                                  fill_value=(y_sorted[0], y_sorted[-1]))
        elif self.interp_type == "akima":
            interpolator = Akima1DInterpolator(x_sorted, y_sorted)
        else:
            raise ValueError(f"Unknown interpolation type: {self.interp_type}")
            
        return interpolator(x)

    @proposal(name='birth', weight=1)
    def birth(self):
        inactive_idx = np.where(~self.state.configuration)[0]
        if len(inactive_idx) == 0:
            return None
            
        idx_to_add = np.random.choice(inactive_idx)
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
        active_idx = np.where(self.state.configuration)[0]
        if len(active_idx) <= self.min_knots:
            return None
            
        idx_to_remove = np.random.choice(active_idx)
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
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = np.random.choice(active_idx)
        
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
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = np.random.choice(active_idx)
        
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
        active_idx = np.where(self.state.configuration)[0]
        idx_to_change = np.random.choice(active_idx)
        
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

    def _calculate_birth_ratio(self, idx_to_add, new_heights, new_knots):
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
        if height < self.ylow or height > self.yhigh:
            return -np.inf
        return -np.log(self.yrange)

    def ln_likelihood(self, configuration, heights, knots):
        raise NotImplementedError("Subclasses must implement ln_likelihood")

    def sample(self, Niterations, prior_test=False, proposal_cycle=None,
            start_heights=None, start_knots=None):
        """
        Run MCMC sampling with optional starting state
        """
        if proposal_cycle is not None:
            self.proposal_manager.proposals = proposal_cycle
            
        # Set initial state
        if all(x is not None for x in [start_heights, start_knots]):
            self.state = SplineState(
                configuration=self.state.configuration,
                heights=start_heights,
                knots=start_knots
            )
        elif any(x is not None for x in [start_heights, start_knots]):
            raise ValueError("If you provide a starting state, you must provide all components.")
    
    # Continue with existing MCMC logic
    # ...existing code...
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
        log_likelihoods[0] = self.ln_likelihood(
            self.state.configuration,
            self.state.heights,
            self.state.knots
        ) if not prior_test else 0
        
        n_accepted = 0
        
        with tqdm(total=Niterations, desc="Sampling", unit="iter") as pbar:
            for i in range(1, Niterations):
                # Get proposal type
                proposal_type = self.proposal_manager.get_next_proposal()
                proposals_used.append(proposal_type)
                
                # Get proposal
                proposal = self.proposal_manager.proposals[proposal_type]()
                    
                if proposal is None:
                    # No valid proposal, copy previous state
                    configurations[i] = configurations[i-1]
                    heights[i] = heights[i-1]
                    knots[i] = knots[i-1]
                    log_likelihoods[i] = log_likelihoods[i-1]
                    continue
                    
                # Calculate acceptance probability
                if prior_test:
                    log_alpha = proposal.log_ratio
                else:
                    log_alpha = proposal.log_likelihood - log_likelihoods[i-1] + proposal.log_ratio
                    
                log_ratios[i] = proposal.log_ratio

                # Accept/reject
                if np.log(np.random.rand()) < log_alpha:
                    configurations[i] = proposal.new_config
                    heights[i] = proposal.new_heights
                    knots[i] = proposal.new_knots
                    log_likelihoods[i] = proposal.log_likelihood
                    
                    # Update current state
                    self.state = SplineState(
                        configuration=proposal.new_config,
                        heights=proposal.new_heights,
                        knots=proposal.new_knots,
                        log_likelihood=proposal.log_likelihood
                    )
                    n_accepted += 1
                    acceptances[i] = True
                else:
                    configurations[i] = configurations[i-1]
                    heights[i] = heights[i-1]
                    knots[i] = knots[i-1]
                    log_likelihoods[i] = log_likelihoods[i-1]
                
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
            acceptance_rate=n_accepted/(Niterations-1),
            proposals_used=proposals_used,
            acceptances=acceptances,
            log_ratios=log_ratios
        )


class SamplerResults:
    def __init__(self, log_likelihoods, configurations, heights, knots, acceptance_rate, proposals_used, acceptances, log_ratios):
        self.log_likelihoods = log_likelihoods
        self.configurations = configurations
        self.heights = heights
        self.knots = knots
        self.acceptance_rate = acceptance_rate
        self.proposals_used = proposals_used
        self.acceptances = acceptances
        self.log_ratios = log_ratios

    def get_num_knots(self):
        return np.sum(self.configurations, axis=1)

    def return_knot_info(self, offset=0):
        knot_configs = self.configurations  # [offset:, :]
        num_knots = knot_configs.sum(axis=1)
        return knot_configs, num_knots
        
    def return_knot_placements(self, offset=0):
        all_weights = []
        all_bins = []
        for ii in range(self.knots.shape[1]):
            if np.sum(self.configurations.astype(bool)[offset:, ii]) > 0:    
                weights, bins, x = plt.hist(self.knots[self.configurations.astype(bool)[:, ii], ii])
                all_weights.append(weights)
                all_bins.append(bins[:-1])
        #plt.show()
        return all_bins, all_weights

    def return_knot_heights(self, offset=0, toggle=False):
        if toggle:
            knot_heights = []
            for ii in range(self.knots.shape[1]):
                knot_heights.append(10**(self.heights[self.configurations.astype(bool)[:, ii], ii]))
        else: 
            knot_heights = 10**self.heights[offset:, :]
        return knot_heights

    def return_knot_frequencies(self, offset=0, toggle=False):
        temp = []
        for ii in range(self.knots.shape[1]):
            if toggle:
                temp.append(self.knots[self.configurations.astype(bool)[:, ii], ii])
            else:
                temp.append(self.knots[:, ii])
        return temp

    def print_acceptance_rates_by_proposal(self):
        from collections import Counter
        total_counts = Counter(self.proposals_used)
        accepted_counts = Counter()
        
        for i, proposal in enumerate(self.proposals_used):
            if self.acceptances[i]:
                accepted_counts[proposal] += 1
        
        print(accepted_counts)
        for proposal, count in total_counts.items():
            rate = accepted_counts[proposal] / count if count else 0
            print(f"{proposal}: {rate:.3f} {accepted_counts[proposal]}")

class SmoothCurveDataObj(object):
    """
    A data class that can be used with our spline model
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitSmoothCurveModel(BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    def ln_likelihood(self, config, heights, knots):
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
        model = self.evaluate_interp_model(self.data.data_xvals, heights, config, knots)
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))
