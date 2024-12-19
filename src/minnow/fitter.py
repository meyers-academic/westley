import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from scipy.stats import norm
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from loguru import logger



class BaseSplineModel(object):
    def __init__(self, data, N_possible_knots, xrange,
                 height_prior_range, interp_type='linear', log_output=False,
                 log_space_xvals=False, birth_uniform_frac=0.5,
                 min_knots=2, birth_gauss_scalefac=1):
        """
        Params:
        ------
        data : `object`
            Abstract data object that gets passed to the likelihood
            and used in calculating the likelihood
        N_possible_knots : `int`
            number of possible knots
        xrange : `tuple`
            low and high values on the x-axis  (low, high)
            between which we place the N knots.
        height_prior_range : `tuple`
            low and high values for the uniform prior on the heights
            of the knots
        interp_type : `str`
            interpolation type. "linear," "cubic" and "akima" are
            the valid options.
        """
        self.birth_uniform_frac = birth_uniform_frac
        self.birth_gauss_scalefac = birth_gauss_scalefac
        self.data = data
        self.N_possible_knots = N_possible_knots
        self.min_knots = min_knots
        self.xlow = xrange[0] + 1e-3
        self.xhigh = xrange[1]


        self.yhigh = height_prior_range[1]
        self.ylow = height_prior_range[0]

        self.yrange = self.yhigh - self.ylow

        self.interp_type = interp_type
        if log_space_xvals:
            # raise NotImplementedError('log_space_xvals not implemented at the moment')
            base = np.logspace(np.log10(self.xlow), np.log10(self.xhigh), num=self.N_possible_knots + 1)
            self.xlows = base[:-1]
            self.xhighs = base[1:]
            self.available_knots = (self.xlows + self.xhighs) / 2
        else:
            self.deltax = (self.xhigh - self.xlow) / N_possible_knots
            self.xlows = np.arange(N_possible_knots) * self.deltax + self.xlow
            self.xhighs = self.xlows + self.deltax
            self.available_knots = np.linspace(self.xlow + self.deltax / 2, self.xhigh - self.deltax / 2, num=self.N_possible_knots)

        # keeps track of configuration, i.e. what points are turned on
        # or turned off.
        self.configuration = np.ones(self.N_possible_knots, dtype=bool)
        # self.configuration = np.random.randint(0, 2, size=self.N_possible_knots).astype(bool)
        self.current_heights = np.ones(self.N_possible_knots) * (self.yhigh - self.ylow) / 2. + self.ylow
        self.log_output = log_output

        # for handling proposals
        self._proposal_cycle = {
            'birth': 1,
            'death': 1,
            'change_amplitude_prior_draw': 1,
            'change_amplitude_gaussian': 1,
            'change_knot_location': 1
        }
        self._available_proposals = list(self._proposal_cycle.keys())

    @property
    def available_proposals(self):
        """The available_proposals property."""
        return self._available_proposals
    @available_proposals.setter
    def available_proposals(self, value):
        self._available_proposals = value

    @property
    def proposal_cycle(self):
        """proposal cycle"""
        return self._proposal_cycle
    @proposal_cycle.setter
    def proposal_cycle(self, value):
        self._proposal_cycle = value
        self._available_proposals = list(self._proposal_cycle.keys())

    def evaluate_interp_model(self, xvals_to_evaluate, heights, config, knots, log_xvals=False):
        """
        based on the supplied configuration and heights of the knots
        evaluate the model at `xvals_to_evaluate`.
        """
        if log_xvals:
            knots = np.log10(knots)
        if np.sum(config) == 0:
            if self.log_output:
                return -np.inf * np.ones(xvals_to_evaluate.size)
            else:
                return np.zeros(xvals_to_evaluate.size)
        elif np.sum(config) == 1:
            return heights[config].squeeze() * np.ones(xvals_to_evaluate.size)
        elif self.interp_type == 'linear':
            myfunc = interp1d(knots[config], heights[config],
                              fill_value='extrapolate')
        elif self.interp_type == 'cubic':
            myfunc = CubicSpline(knots[config], heights[config],
                                 extrapolate=True)
        elif self.interp_type == 'akima':
            myfunc = Akima1DInterpolator(knots[config], heights[config])
            return myfunc(xvals_to_evaluate, extrapolate=True)
        else:
            raise ValueError('available spline types are "linear," "cubic" and "akima"')
        return myfunc(xvals_to_evaluate)

    @abstractmethod
    def ln_likelihood(self, config, heights, knot_locations):
        """
        You will need to implement this yourself. It will take the model, and put it into whatever space
        it needs to be in to calculate your likelihood, and then calculate the likelihood.
        """
        pass

    def birth(self):
        if np.sum(self.configuration) == self.N_possible_knots:
            return (-np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots)
        else:
            idx_to_add = np.random.choice(np.where(~self.configuration)[0])
            new_heights = deepcopy(self.current_heights)
            new_config = deepcopy(self.configuration)
            new_config[idx_to_add] = True

        randnum = np.random.rand()

        # random choice of knot location within bounds
        new_knots = self.available_knots
        new_knots[idx_to_add] = np.random.rand() * (self.xhighs[idx_to_add] - self.xlows[idx_to_add]) + self.xlows[idx_to_add]

        # proposal height
        height_from_model = self.evaluate_interp_model(new_knots[idx_to_add],
                                                       self.current_heights, self.configuration, self.available_knots)
        if randnum < self.birth_uniform_frac:
            # uniform draw
            new_heights[idx_to_add] = np.random.rand() * (self.yhigh - self.ylow) + self.ylow
        else:
            # gaussian draw around height
            new_heights[idx_to_add] = norm.rvs(loc=height_from_model, scale=self.birth_gauss_scalefac, size=1)


        log_qx = 0

        log_qy = np.log(self.birth_uniform_frac / self.yrange + \
                        (1 - self.birth_uniform_frac) * norm.pdf(new_heights[idx_to_add], loc=height_from_model,
                                                                    scale=self.birth_gauss_scalefac))

        log_px = 0

        # log_py = self.get_height_log_prior(new_heights[idx_to_add])
        log_py = self.get_height_log_prior(new_heights[idx_to_add]) # + self.get_width_log_prior(new_knots[idx_to_add], idx_to_add)

        new_ll = self.ln_likelihood(new_config, new_heights, new_knots)

        return new_ll, (log_py - log_px) + (log_qx - log_qy), new_config, new_heights, new_knots

    def death(self, specific_idx=None):
        """
        propose to "turn off" one of the current knots that are turned on.
        """
        if np.sum(self.configuration) == self.min_knots:
            return (-np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots)
        else:
            # pick one to turn off
            idx_to_remove = np.random.choice(np.where(self.configuration)[0])
            new_heights = deepcopy(self.current_heights)
            new_config = deepcopy(self.configuration)

            # turn it off
            if specific_idx is None:
                new_config[idx_to_remove] = False
            else:
                idx_to_remove = specific_idx
                new_config[idx_to_remove] = False


            # Find mean of the Gaussian we would have proposed from
            height_from_model = self.evaluate_interp_model(self.available_knots[idx_to_remove],
                                                           self.current_heights, new_config, self.available_knots)

            log_qx = np.log(self.birth_uniform_frac / self.yrange + \
                              (1 - self.birth_uniform_frac) * norm.pdf(self.current_heights[idx_to_remove],
                                                                          loc=height_from_model,
                                                                          scale=self.birth_gauss_scalefac))
            log_qy = 0

            log_px = self.get_height_log_prior(self.current_heights[idx_to_remove]) # + self.get_width_log_prior(self.available_knots[idx_to_remove], idx_to_remove)

            log_py = 0

            new_ll = self.ln_likelihood(new_config, self.current_heights, self.available_knots)

            return new_ll, (log_py - log_px) + (log_qx - log_qy), new_config, new_heights, self.available_knots

    def change_amplitude_gaussian(self):
        """
        Pick one of the knots that are turned
        on and propose to change
        its height by some small amount.
        """
        # random point to turn on
        if np.sum(self.configuration) == 0:
            return -np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots
        idx_to_change = np.random.choice(np.where(self.configuration)[0])

        # draw a "scale" factor between 1/10 and 1/3 of prior range
        scalefac = (self.yhigh - self.ylow) * (np.random.rand() * (1/10 - 1/100) + 1/100)

        # propose to jump an amount given by zero-mean Gaussian with standard
        # deviation given by scalefac above
        new_heights = deepcopy(self.current_heights)
        new_heights[idx_to_change] = self.current_heights[idx_to_change] + np.random.randn() * scalefac

        new_ll = self.ln_likelihood(self.configuration, new_heights, self.available_knots)
        prior_change = self.get_height_log_prior(new_heights[idx_to_change])
        if prior_change != -np.inf:
            prior_change = 0

        return new_ll, prior_change, self.configuration, new_heights, self.available_knots


    def get_height_log_prior(self, height):
        if self.ylow <= height <= self.yhigh:
            return -np.log(self.yrange)
        return -np.inf

    def get_width_log_prior(self, val, idx):
        if self.xlows[idx] <= val <= self.xhighs[idx]:
            return -np.log(self.xhighs[idx] - self.xlows[idx])
        return -np.inf

    def change_amplitude_prior_draw(self):
        """
        choose one of the knots that are turned on and propose
        a new height that is drawn from the prior.
        """

        if np.sum(self.configuration) == 0:
            return -np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots
        # random point to change amplitude
        idx_to_change = np.random.choice(np.where(self.configuration)[0])
        # propose draw from prior
        new_heights = deepcopy(self.current_heights)
        new_heights[idx_to_change] = (self.yhigh - self.ylow) * np.random.rand() + self.ylow

        new_ll = self.ln_likelihood(self.configuration, new_heights, self.available_knots)

        prior_change = self.get_height_log_prior(new_heights[idx_to_change])
        if prior_change != -np.inf:
            prior_change = 0
        return new_ll, prior_change, self.configuration, new_heights, self.available_knots

    def change_knot_location(self):
        """change the location of one of the knots that are turned on
        """
        if np.sum(self.configuration) == 0:
            return -np.inf, -np.inf, self.configuration, self.current_heights, self.available_knots
        # find a knot that is turned on and change its location
        idx_to_change = np.random.choice(np.where(self.configuration)[0])
        new_knots = deepcopy(self.available_knots)
        new_knots[idx_to_change] = np.random.rand() * (self.xhighs[idx_to_change] - self.xlows[idx_to_change]) + self.xlows[idx_to_change]
        # new_ll = self.ln_likelihood(self.configuration, self.current_heights, new_knots)
        new_ll = self.ln_likelihood(self.configuration, self.current_heights, new_knots)
        prior_change = self.get_width_log_prior(new_knots[idx_to_change], idx_to_change)
        if prior_change != -np.inf:
            prior_change = 0
        return new_ll, prior_change, self.configuration, self.current_heights, new_knots




    def sample(self, Niterations, proposal_cycle=None, prior_test=False,
               start_config=None, start_heights=None, temperature=1, start_knots=None):
        """
        Run RJMCMC sampler

        Parameters:
        -----------
        Niterations : `int`
            Number of MCMC samples
        proposal_weights : `list` or `np.ndarray`, optional
            list of weights for proposals. In order they are currently:
                [birth, death, prior draw, gaussian]
        prior_test : `bool`, optional, default=False
            If True, it sets likelihood to 0 always and
            samples only from the prior. Vital to check that
            when adding new proposals you still get back the prior.
        start_config : `np.ndarray`, optional, default=None
            Array of boolean values that turn on or off certain knots.
        start_heights: `np.ndarray`, optional, default=None
            Array of starting heights for the knots.

        Returns:
        --------
        results : `SamplerResults`
            sampler results object that contains configurations,
            heights, acceptances, likelihoods, and proposal types
            for each MCMC step.

            configurations = (Nsamples x Nspline points), for example.
        """

        if start_config is not None:
            if np.size(start_config) == self.N_possible_knots:
                self.configuration = start_config
            else:
                print('Start config you entered is not compatible...starting with all knots on')
        if start_heights is not None:
            if np.size(start_heights) == self.N_possible_knots:
                self.current_heights = start_heights
            else:
                print('Start heights you entered is not compatible...starting with heights at zero')
        if start_knots is not None:
            if np.size(start_knots) == self.N_possible_knots:
                self.available_knots = start_knots
            else:
                print('Start heights you entered is not compatible...starting with heights at zero')
        configurations = np.zeros((Niterations, self.N_possible_knots))
        heights = np.zeros((Niterations, self.N_possible_knots))
        knots = np.zeros((Niterations, self.N_possible_knots))

        acceptances = np.zeros(Niterations, dtype=bool)
        move_types = np.zeros(Niterations, dtype=int)

        lls = np.zeros(Niterations, dtype=float)

        current_ll = self.ln_likelihood(self.configuration, self.current_heights, self.available_knots)
        print(f'starting ll: {current_ll}')
        if np.isnan(current_ll):
            current_ll = -np.inf

        # list of functions for proposals
        if proposal_cycle is not None:
            self.proposal_cycle = proposal_cycle
        proposals = [getattr(self, p) for p in self.available_proposals]
        proposal_weights = [self.proposal_cycle[p] for p in self.available_proposals]
        logger.info(f"Proposal Cycle: {self.proposal_cycle}")
        if 'birth' in self.available_proposals and 'death' in self.available_proposals:
            birth_to_death_ratio = self.proposal_cycle['birth'] / self.proposal_cycle['death']

        for ii in tqdm(range(Niterations)):
            # choose proposal
            myval = np.random.rand()
            # choose proposal function with weights that were specified
            proposal_idx = np.random.choice(np.arange(len(proposals)), p=np.array(proposal_weights) / np.sum(proposal_weights))

            # get proposed points
            tmp = proposals[proposal_idx]()
            # print('prop idx', proposal_idx)
            proposed_ll, proposed_logR, proposed_config, proposed_heights, proposed_knots = tmp
            if prior_test:
                proposed_ll = 0

            # need to handle ratio of probabilities of birth vs. death proposals
            # because prob(n -> n+1) would be different from prob(n+1 -> n)
            # if ratio with which we make proposals is different.
            # this is subtle...usually the ratio of *which* proposals you choose
            # doesn't matter
            if self.available_proposals[proposal_idx] == 'birth':
                q = -np.log(birth_to_death_ratio)
            elif self.available_proposals[proposal_idx] == 'death':
                q = np.log(birth_to_death_ratio)
            else:
                q = 0

            hastings_ratio = min(np.log(1), (proposed_ll - current_ll) * 1/temperature + proposed_logR + q)
            compare_val = np.log(np.random.rand())

            # print('heights', proposed_heights, self.current_heights)
            # print('ll', proposed_ll, current_ll)
            if compare_val < hastings_ratio:
                self.configuration = proposed_config
                self.current_heights = proposed_heights
                self.available_knots = proposed_knots
                current_ll = proposed_ll
                acc = True
            else:
                acc = False

            configurations[ii] = self.configuration
            heights[ii] = self.current_heights
            knots[ii] = self.available_knots
            acceptances[ii] = acc
            move_types[ii] = proposal_idx
            lls[ii] = current_ll

        return SamplerResults(acceptances, configurations, heights, knots, lls, move_types)


class SamplerResults(object):
    def __init__(self, acceptances, configurations, heights, knots, lls, move_types):
        self.acceptances = acceptances
        self.configurations = configurations
        self.heights = heights
        self.lls = lls
        self.move_types = move_types
        self.knots = knots

    def get_num_knots(self):
        return np.sum(self.configurations, axis=1)


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
