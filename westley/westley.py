import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d, CubicSpline
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

class SingleBaselineInterpolatingModel(object):
    """
    Parameters:
    -----------
    cross_corr_data : numpy.ndarray
        cross correlation data
    cross_corr_uncertainty : numpy.ndarray
        cross correlation data uncertainty
    cross_corr_frequencies : numpy.ndarray
        cross correlation frequencies
    """
    def __init__(self, cross_corr_data, cross_corr_sigma, cross_corr_frequencies, min_knots=1, max_knots=40,
                 amin=1e-12, amax=1e-7, baseline='HL', model_type='cubic'):
        self.cross_corr_data = cross_corr_data
        self.cross_corr_sigma = cross_corr_sigma
        self.cross_corr_frequencies = cross_corr_frequencies
        self.min_knots = min_knots
        self.max_knots = max_knots
        self.amin = amin
        self.amax = amax
        self.baseline = baseline
        self.model_type = model_type

        self.available_knots = np.atleast_1d(np.linspace(min(cross_corr_frequencies), max(cross_corr_frequencies), num=max_knots))
        if np.size(self.available_knots) > 1:
            self.delta_x = np.diff(self.available_knots)[0]
        else:
            self.delta_x = max(cross_corr_frequencies) - min(cross_corr_frequencies)

    def evaluate_model(self, knots, knot_amplitudes, config, model_type='cubic'):

        # one node turned on
        if np.sum(config) == 1:
            return np.ones(self.cross_corr_frequencies.size) * knot_amplitudes[config]
        # no nodes turned on
        elif np.sum(config) == 0:
            return np.zeros(self.cross_corr_frequencies.size)
        # cubic spline model
        elif model_type == 'cubic':
            interpolator = CubicSpline(knots[config],
                                       knot_amplitudes[config], extrapolate=True)
            model = interpolator(self.cross_corr_frequencies)
        # piecewise power law model
        elif model_type == 'pp':
            interpolator = interp1d(np.log10(knots[config]),
                                    np.log10(knot_amplitudes[config]), fill_value="extrapolate")
            model = 10**interpolator(np.log10(self.cross_corr_frequencies))

        return model


class MultiBaselineInterpolatingModel(object):
    def __init__(self, single_baseline_list,
                 common_signal_min_knots=1, common_signal_max_knots=40,
                 common_signal_amax=1e-7, common_signal_amin=1e-12,
                 common_model_type='cubic'):
        self.common_signal_min_knots = common_signal_min_knots
        self.common_signal_max_knots = common_signal_max_knots
        self.common_signal_amin = common_signal_amin
        self.common_signal_amax = common_signal_amax
        self.baseline_dict = {bl.baseline: bl for bl in single_baseline_list}
        self.baselines = [bl.baseline for bl in single_baseline_list]

        minf = min([min(bl.cross_corr_frequencies) for bl in single_baseline_list])
        maxf = max([max(bl.cross_corr_frequencies) for bl in single_baseline_list])

        self.common_signal_available_knots = np.atleast_1d(np.linspace(minf, maxf, num=common_signal_max_knots))
        if np.size(self.common_signal_available_knots) > 1:
            self.common_signal_delta_x = np.diff(self.common_signal_available_knots)[0]
        else:
            self.common_signal_delta_x = maxf - minf
        # self.common_signal_configuration = np.atleast_1d(np.ones(common_signal_max_knots, dtype=bool) )
        self.common_model_type = common_model_type

        # dicts for individual baselines
        # self.individual_knot_amplitudes = starting_knot_amplitudes_dict

    def evaluate_common_model(self, baseline, common_knots, common_knot_amplitudes, config):
        # config = self.common_signal_configuration
        # one node turned on

        if np.sum(config) == 1:
            return np.ones(baseline.cross_corr_frequencies.size) * common_knot_amplitudes[config]
        # no nodes turned on
        elif np.sum(config) == 0:
            return np.zeros(baseline.cross_corr_frequencies.size)
        # cubic spline model
        elif self.common_model_type == 'cubic':
            interpolator = CubicSpline(common_knots[config],
                                       common_knot_amplitudes[config], extrapolate=True)
            model = interpolator(baseline.cross_corr_frequencies)
        # piecewise power law model
        elif self.common_model_type == 'pp':
            interpolator = interp1d(np.log10(common_knots[config]),
                                    np.log10(common_knot_amplitudes[config]), fill_value="extrapolate")
            model = 10**interpolator(np.log10(baseline.cross_corr_frequencies))

        return model

    def evaluate_model(self, individual_knots, individual_knot_amplitudes, individual_config, common_knots, common_knot_amplitudes, common_config):
        individual_models = {}
        common_models = {}
        for bl in self.baseline_dict:
            individual_models[bl] = self.baseline_dict[bl].evaluate_model(individual_knots[bl],
                                                                          individual_knot_amplitudes[bl],
                                                                          individual_config[bl],
                                                                          self.baseline_dict[bl].model_type)
            # evaluate common model at frequencies of this baseline
            common_models[bl] = self.evaluate_common_model(self.baseline_dict[bl], common_knots, common_knot_amplitudes, common_config)
        return individual_models, common_models

    def log_likelihood(self, individual_knots, individual_knot_amplitudes, individual_config, common_knots, common_knot_amplitudes, common_config):
        imodels, cmodels = self.evaluate_model(individual_knots, individual_knot_amplitudes, individual_config,
                                               common_knots, common_knot_amplitudes, common_config)
        ll = 0
        for bl in imodels:
            # model for baseline is individual model + common model
            tmpmodel = imodels[bl] + cmodels[bl]
            ll += np.sum(norm.logpdf(self.baseline_dict[bl].cross_corr_data,
                                     loc=tmpmodel,
                                     scale=self.baseline_dict[bl].cross_corr_sigma))
        return ll


class Sampler(object):
    def __init__(self, mbim):
        self.mbim = mbim
        # initialize individual amplitudes
        # based on data
        self.individual_amplitudes = {}
        self.common_amplitudes = np.atleast_1d(np.zeros(self.mbim.common_signal_max_knots))

        self.individual_knots = {bl: self.mbim.baseline_dict[bl].available_knots for bl in self.mbim.baselines}
        self.common_knots = self.mbim.common_signal_available_knots.copy()


        # set individual configurations to start
        self.individual_config= {mbim.baseline_dict[bl].baseline: np.atleast_1d(np.zeros(mbim.baseline_dict[bl].max_knots, dtype=bool)) for bl in mbim.baseline_dict}
        self.common_config = np.atleast_1d(np.ones(self.mbim.common_signal_max_knots, dtype=bool))

        # set individual amplitudes
        for bl in mbim.baseline_dict:
            if self.mbim.baseline_dict[bl].max_knots > 0:
                # initialize
                self.individual_amplitudes[bl] = np.atleast_1d(np.zeros(self.mbim.baseline_dict[bl].available_knots.size))

                # set
                for ii, ff in enumerate(self.mbim.baseline_dict[bl].available_knots):
                    # find argument of frequency closest to knot
                    arg = np.argmin(np.abs(self.mbim.baseline_dict[bl].cross_corr_frequencies - ff))
                    self.individual_amplitudes[bl][ii] = np.abs(self.mbim.baseline_dict[bl].cross_corr_data[arg])
            else:
                self.individual_amplitudes[bl] = np.array([])

        # set common amplitudes by taking average of closest point across
        # data for all three baselines
        for bl in mbim.baseline_dict:
            for ii, ff in enumerate(self.mbim.common_signal_available_knots):
                # find argument of frequency closest to knot
                arg = np.argmin(np.abs(self.mbim.baseline_dict[bl].cross_corr_frequencies - ff))
                self.common_amplitudes[ii] += np.abs(self.mbim.baseline_dict[bl].cross_corr_data[arg])
        self.common_amplitudes *= 1 / np.size(list(self.mbim.baseline_dict.keys()))


    def propose_birth_move(self):
        # randomly choose whether to update individual signals
        # or common signal. 0 == common, otherwise, baseline
        myval = np.random.randint(0, len(self.mbim.baselines) + 1)
        # common signal birth move
        if myval == 0:
            if np.sum(self.common_config) == self.mbim.common_signal_max_knots:
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)

            # choose index to add from list of knots that are turned off
            idx_to_add = np.random.choice(np.where(~self.common_config)[0])
            new_amps = deepcopy(self.common_amplitudes)
            new_config = deepcopy(self.common_config)

            # random log-uniform draw from common signal prior
            new_amps[idx_to_add] = 10**(np.random.rand(1) * (np.log10(self.mbim.common_signal_amax) - np.log10(self.mbim.common_signal_amin)) + np.log10(self.mbim.common_signal_amin))
            new_config[idx_to_add] = True

            # proposal ratio and log like
            Rval = 1
            new_ll = self.mbim.log_likelihood(self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, new_amps, new_config)
            return (new_ll, np.log(Rval),
                    self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, new_amps, new_config)
        else:
            # get the baseline we're updating
            bl = self.mbim.baselines[myval - 1]
            # check if we can add anything
            if np.sum(self.individual_config[bl]) == self.mbim.baseline_dict[bl].max_knots:
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)
            idx_to_add = np.random.choice(np.where(~self.individual_config[bl])[0])

            # amp and config dicts to change
            amp_dict_copy = deepcopy(self.individual_amplitudes)
            config_dict_copy = deepcopy(self.individual_config)

            # update with new values
            amp_dict_copy[bl][idx_to_add] = 10**(np.random.rand(1) * (np.log10(self.mbim.baseline_dict[bl].amax) - np.log10(self.mbim.baseline_dict[bl].amin)) + np.log10(self.mbim.baseline_dict[bl].amin))
            config_dict_copy[bl][idx_to_add] = True
            # new likelihood
            new_ll = self.mbim.log_likelihood(self.individual_knots, amp_dict_copy, config_dict_copy,
                                              self.common_knots, self.common_amplitudes, self.common_config)
            Rval = 1

            return (new_ll, np.log(Rval), self.individual_knots, amp_dict_copy, config_dict_copy,
                    self.common_knots, self.common_amplitudes, self.common_config)

    def propose_death_move(self):
        # randomly choose whether to update individual signals
        # or common signal. 0 == common, otherwise, baseline
        myval = np.random.randint(0, len(self.mbim.baselines) + 1)

        # common signal death move
        if myval == 0:
            if np.sum(self.common_config) == self.mbim.common_signal_min_knots:
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, self.common_amplitudes, self.common_config)

            # choose index to add from list of knots that are turned off
            idx_to_remove = np.random.choice(np.where(self.common_config)[0])
            new_amps = deepcopy(self.common_amplitudes)
            new_config = deepcopy(self.common_config)

            # random log-uniform draw from common signal prior
            new_config[idx_to_remove] = False

            # proposal ratio and log like
            Rval = 1
            new_ll = self.mbim.log_likelihood(self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, new_amps, new_config)
            return (new_ll, np.log(Rval), self.individual_knots, self.individual_amplitudes, self.individual_config,
                    self.common_knots, new_amps, new_config)

        # individual baseline death proposal
        else:
            # get the baseline we're updating
            bl = self.mbim.baselines[myval - 1]
            # check if we can add anything
            if np.sum(self.individual_config[bl]) == self.mbim.baseline_dict[bl].min_knots:
                return (-np.inf, -np.inf, self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)

            # pick a control point to remove
            idx_to_remove = np.random.choice(np.where(self.individual_config[bl])[0])

            # amp and config dicts to change
            amp_dict_copy = deepcopy(self.individual_amplitudes)
            config_dict_copy = deepcopy(self.individual_config)

            # update with new values
            config_dict_copy[bl][idx_to_remove] = False

            # new likelihood
            new_ll = self.mbim.log_likelihood(self.individual_knots, amp_dict_copy, config_dict_copy,
                                              self.common_knots, self.common_amplitudes, self.common_config)
            Rval = 1

            return (new_ll, np.log(Rval), self.individual_knots, amp_dict_copy, config_dict_copy,
                    self.common_knots, self.common_amplitudes, self.common_config)

    def propose_move_control_point(self):
        # randomly choose whether to update individual signals
        # or common signal. 0 == common, otherwise, baseline
        myval = np.random.randint(0, len(self.mbim.baselines) + 1)

        # common signal death move
        if myval == 0:
            if np.sum(self.common_config) == self.mbim.common_signal_min_knots:
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, self.common_amplitudes, self.common_config)
            # choose index to add from list of knots that are turned off
            idx_to_move = np.random.choice(np.where(self.common_config)[0])

            new_amps = deepcopy(self.common_amplitudes)
            new_config = deepcopy(self.common_config)
            new_config[idx_to_move] = False
            idx_to_move_to = np.random.choice(np.where(~new_config)[0]) 
            new_config[idx_to_move_to] = True

            # random log-uniform draw from common signal prior
            new_amps[idx_to_move_to] = 10**(np.random.rand(1) * (np.log10(self.mbim.common_signal_amax) - np.log10(self.mbim.common_signal_amin)) + np.log10(self.mbim.common_signal_amin)) 

            # proposal ratio and log like
            Rval = 1
            new_ll = self.mbim.log_likelihood(self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, new_amps, new_config)
            return (new_ll, np.log(Rval),
                    self.individual_knots, self.individual_amplitudes, self.individual_config, self.common_knots, new_amps, new_config)
        # individual baseline death proposal
        else:
            # get the baseline we're updating
            bl = self.mbim.baselines[myval - 1]
            if np.sum(self.individual_config[bl]) == self.mbim.baseline_dict[bl].min_knots:
                return (-np.inf, -np.inf, self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)

            amp_dict_copy = deepcopy(self.individual_amplitudes)
            config_dict_copy = deepcopy(self.individual_config)

            # get rid of one point
            idx_to_move = np.random.choice(np.where(self.individual_config[bl])[0])
            config_dict_copy[bl][idx_to_move] = False
            # birth another at the same time
            # print(config_dict_copy[bl])
            idx_to_move_to = np.random.choice(np.where(~config_dict_copy[bl])[0])
            config_dict_copy[bl][idx_to_move_to] = True
            amp_dict_copy[bl][idx_to_move_to] = 10**(np.random.rand(1) * (np.log10(self.mbim.baseline_dict[bl].amax) - np.log10(self.mbim.baseline_dict[bl].amin)) + np.log10(self.mbim.baseline_dict[bl].amin))
            # new likelihood
            new_ll = self.mbim.log_likelihood(self.individual_knots, amp_dict_copy, config_dict_copy,
                                              self.common_knots, self.common_amplitudes, self.common_config)
            Rval = 1

            return (new_ll, np.log(Rval), self.individual_knots, amp_dict_copy, config_dict_copy,
                    self.common_knots, self.common_amplitudes, self.common_config)

    def propose_amplitude_jump(self, scale=0.35):
        myval = np.random.randint(0, len(self.mbim.baselines) + 1)

        # update amplitude for common signals
        if myval == 0:
            if np.sum(self.common_config) == 0:
                # dummy for keeping random number consistency
                return (-np.inf, -np.inf, self.individual_knots, self.individual_amplitudes,
                        self.individual_config, self.common_knots, self.common_amplitudes, self.common_config)

            idx = np.random.choice(np.where(self.common_config)[0])
            common_amps = deepcopy(self.common_amplitudes)
            common_amps[idx] += common_amps[idx] * np.random.randn(1) * scale
            # check prior
            if np.any(common_amps < self.mbim.common_signal_amin) or np.any(common_amps > self.mbim.common_signal_amax):
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)

            new_ll = self.mbim.log_likelihood(self.individual_knots, self.individual_amplitudes, self.individual_config,
                                                self.common_knots, common_amps, self.common_config)
            Rval = 1
            return (new_ll, np.log(Rval), self.individual_knots, self.individual_amplitudes, self.individual_config,
                    self.common_knots, common_amps, self.common_config)
        else:
            bl = self.mbim.baselines[myval - 1]
            if np.sum(self.individual_config[bl]) == 0:
                return (-np.inf, -np.inf, self.individual_knots, self.individual_amplitudes,
                        self.individual_config, self.common_knots, self.common_amplitudes, self.common_config)

            new_ind_amps = deepcopy(self.individual_amplitudes)
            idx = np.random.choice(np.where(self.individual_config[bl])[0])
            # add a random jump
            new_ind_amps[bl][idx] += scale * new_ind_amps[bl][idx] * np.random.randn(1)

            if np.any(new_ind_amps[bl] < self.mbim.baseline_dict[bl].amin) or np.any(new_ind_amps[bl] > self.mbim.baseline_dict[bl].amax):
                return (-np.inf, -np.inf,
                        self.individual_knots, self.individual_amplitudes, self.individual_config,
                        self.common_knots, self.common_amplitudes, self.common_config)
            new_ll = self.mbim.log_likelihood(self.individual_knots, new_ind_amps, self.individual_config,
                                              self.common_knots, self.common_amplitudes, self.common_config)
            Rval = 1
            return (new_ll, np.log(Rval), self.individual_knots, new_ind_amps, self.individual_config,
                    self.common_knots, self.common_amplitudes, self.common_config)

    def sample(self, niter=int(1e5), thin=10, burn=0.5):
        np.random.seed(1)
        burn = int(niter * burn)
        Nsave = int(niter / thin)
        burn_thin = int(burn / thin)
        individual_amps = {bl: np.zeros((Nsave - burn_thin, self.mbim.baseline_dict[bl].max_knots)) for bl in self.mbim.baselines}
        individual_configs = {bl: np.zeros((Nsave - burn_thin, self.mbim.baseline_dict[bl].max_knots), dtype=bool) for bl in self.mbim.baselines}
        individual_knots = {bl: np.zeros((Nsave - burn_thin, self.mbim.baseline_dict[bl].max_knots)) for bl in self.mbim.baselines}

        common_knots = np.zeros((Nsave - burn_thin, self.mbim.common_signal_max_knots))
        common_config = np.zeros((Nsave - burn_thin, self.mbim.common_signal_max_knots), dtype=bool)
        common_amps = np.zeros((Nsave - burn_thin, self.mbim.common_signal_max_knots))

        acceptances = np.zeros(Nsave - burn_thin, dtype=bool)
        move_types = np.zeros(Nsave - burn_thin)
        
        lls = np.zeros(Nsave - burn_thin)

        self.current_ll = -np.inf

        for ii in tqdm(range(niter)):
            myval = np.random.rand()
            # if ii < niter:
            #     myval = 0.8
            if ii == int(niter / 4):
                print("adding birth and death proposals")
            if myval < 0.5:
                ll, logR, ik, ia, ic, ck, ca, cc = self.propose_birth_move()
                mt = 1
            elif myval >=  0.5 and myval < 1:
                ll, logR, ik, ia, ic, ck, ca, cc = self.propose_death_move()
                mt = 2
            elif myval >= 0  and myval < 0: 
                ll, logR, ik, ia, ic, ck, ca, cc = self.propose_move_control_point()
                mt = 3
            elif myval <= 0:
                ll, logR, ik, ia, ic, ck, ca, cc = self.propose_amplitude_jump()
                mt = 4

            hastings_ratio = min(np.log(1), ll - self.current_ll + logR) 
            compare_val = np.log(np.random.rand())

            if compare_val < hastings_ratio:
                self.individual_knots = ik.copy()
                self.individual_amplitudes = ia.copy()
                self.individual_config = ic.copy()

                self.common_knots = ck.copy()
                self.common_amplitudes = ca.copy()
                self.common_config = cc.copy()
                self.current_ll = ll
                acc = True

            # reject
            else:
                acc = False
            # bookkeeping
            if ii % thin == 0 and ii >= burn:
                idx = int((ii - burn) / thin)
                acceptances[idx] = acc

                move_types[idx] = mt

                lls[idx] = self.current_ll

                common_amps[idx] = self.common_amplitudes.copy()
                common_config[idx] = self.common_config.copy()
                common_knots[idx] = self.common_knots.copy()

                for bl in self.mbim.baselines:
                    individual_knots[bl][idx] = self.individual_knots[bl].copy()
                    individual_amps[bl][idx] = self.individual_amplitudes[bl].copy()
                    individual_configs[bl][idx] = self.individual_config[bl].copy()

        return WestleyResults(acceptances, lls, move_types, individual_knots, individual_amps, individual_configs, common_knots, common_amps, common_config)


def get_dicts_from_idx(ik, ia, ic, idx):
    mydict_knots = {}
    mydict_amps = {}
    mydict_config = {}
    for bl in ik.keys():
        mydict_knots[bl] = ik[bl][idx]
        mydict_amps[bl] = ia[bl][idx]
        mydict_config[bl] = ic[bl][idx]
    return mydict_knots, mydict_amps, mydict_config

class WestleyResults(object):
    def __init__(self, acceptances, lls, move_types, individual_knots, individual_amps, individual_configs, common_knots, common_amps, common_config):
        self.acceptances = acceptances
        self.lls = lls
        self.move_types = move_types
        self.individual_knots = individual_knots
        self.individual_amps = individual_amps
        self.individual_configs = individual_configs
        self.common_knots = common_knots
        self.common_amps = common_amps
        self.common_config = common_config

        self.Nsamples = np.size(lls)

    def plot_model_on_data(self, mbim, nsamples, alpha_weight=0.01):
        print(len(mbim.baselines))
        fig, axs = plt.subplots(1, len(mbim.baselines), figsize=(6*len(mbim.baselines), 4))
        if len(mbim.baselines) == 1:
            axs = [axs]
        keys = list(mbim.baseline_dict.keys())
        for ii in range(nsamples):
            idx = np.random.choice(self.common_config.shape[0])
            mylist = get_dicts_from_idx(self.individual_knots, self.individual_amps, self.individual_configs, idx)
            im, cm = mbim.evaluate_model(mylist[0], mylist[1], mylist[2], self.common_knots[idx], self.common_amps[idx], self.common_config[idx])

            for jj, blkey in enumerate(keys):
                axs[jj].plot(mbim.baseline_dict[blkey].cross_corr_frequencies, np.array(im[blkey]) + np.array(cm[blkey]), alpha=alpha_weight)
            # plt.plot(frequencies, im['HL'], label='individual')
            # plt.plot(frequencies, cm['HL'], label='combined')

        for jj, blkey in enumerate(keys):
            axs[jj].errorbar(mbim.baseline_dict[blkey].cross_corr_frequencies, mbim.baseline_dict[blkey].cross_corr_data,
                             yerr=mbim.baseline_dict[blkey].cross_corr_sigma, fmt='o', alpha=0.5, zorder=-1000)
            # axs[ii].legend()
            axs[jj].set_xlabel('Frequency [Hz]')
            axs[jj].set_ylabel('$\Omega_{gw}(f)$')
        return fig
