import unittest
import numpy as np
from westley import SingleBaselineInterpolatingModel, MultiBaselineInterpolatingModel


class TestSingleBaselineInterpolatingModel(unittest.TestCase):
    def setUp(self):
        self.cross_corr_data = np.zeros(10)
        self.cross_corr_frequencies = np.linspace(10, 100, num=10)
        self.cross_corr_sigmas = np.ones(10)
        self.min_knots = 1
        self.max_knots = 40
        self.baseline = 'HL'
        self.model_type = 'cubic'
        self.knots = np.linspace(10, 100, num=40)

    def test_initialization(self):
        sbim =  SingleBaselineInterpolatingModel(self.cross_corr_data,
                                                 self.cross_corr_sigmas, 
                                                 self.cross_corr_frequencies,
                                                 self.min_knots,
                                                 self.max_knots,
                                                 baseline=self.baseline,
                                                 model_type='cubic')

    def test_evaluate_model(self):
        sbim =  SingleBaselineInterpolatingModel(self.cross_corr_data,
                                                 self.cross_corr_sigmas, 
                                                 self.cross_corr_frequencies,
                                                 self.min_knots,
                                                 self.max_knots,
                                                 baseline=self.baseline,
                                                 model_type='cubic')
        # all on, cubic and pp
        model = sbim.evaluate_model(self.knots, np.ones(self.max_knots), np.ones(self.max_knots, dtype=bool), model_type='cubic')
        model = sbim.evaluate_model(self.knots, np.ones(self.max_knots), np.ones(self.max_knots, dtype=bool), model_type='pp')
        self.assertTrue(np.all(model == np.ones(self.cross_corr_frequencies.size)))
        # all off
        model = sbim.evaluate_model(self.knots, 10*np.ones(self.max_knots), np.zeros(self.max_knots, dtype=bool), model_type='cubic')
        model = sbim.evaluate_model(self.knots, 10*np.ones(self.max_knots), np.zeros(self.max_knots, dtype=bool), model_type='pp')
        self.assertTrue(np.all(model == np.zeros(self.cross_corr_frequencies.size)))
        # one on (constant model)
        config = np.zeros(self.max_knots, dtype=bool)
        config[0] = 1
        myval = 1.234
        model = sbim.evaluate_model(self.knots, myval * np.ones(self.max_knots), config, model_type='cubic')
        model = sbim.evaluate_model(self.knots, myval * np.ones(self.max_knots), config, model_type='pp')
        self.assertTrue(np.all(model == np.ones(self.cross_corr_frequencies.size) * myval))


class TestMultiBaselineInterpolatingModel(unittest.TestCase):
    def setUp(self):
        sbim1 = SingleBaselineInterpolatingModel(np.zeros(10), np.ones(10), np.linspace(10, 30, num=10))
        sbim2 = SingleBaselineInterpolatingModel(np.zeros(10), np.ones(10), np.linspace(11, 31, num=10), baseline='HV')
        sbim3 = SingleBaselineInterpolatingModel(np.zeros(10), np.ones(10), np.linspace(12, 32, num=10), baseline='LV')
        self.baseline_dict = {'HL': sbim1, 'HV': sbim2,'LV': sbim3}
        self.baseline_list = [sbim1, sbim2, sbim3]
        self.starting_knot_amplitudes = {}
        self.starting_knot_values = {}
        for bl in self.baseline_dict:
            self.starting_knot_amplitudes[bl] = np.random.rand(self.baseline_dict[bl].available_knots.size)
        for bl in self.baseline_dict:
            self.starting_knot_values[bl] = self.baseline_dict[bl].available_knots.copy()


    def test_initialization(self):
        mbim = MultiBaselineInterpolatingModel(self.baseline_list , self.starting_knot_amplitudes)
        individual_models = {}
        individual_knots = {}
        for bl in self.baseline_dict:
            individual_models[bl] = np.random.rand(mbim.baseline_dict[bl].available_knots.size)
            individual_knots[bl] = mbim.baseline_dict[bl].available_knots.copy()
        common_knots = mbim.common_signal_available_knots.copy()
        mbim.evaluate_model(individual_knots, individual_models, common_knots, np.random.randn(mbim.common_signal_available_knots.size))
        ll = mbim.log_likelihood(individual_knots, individual_models, common_knots, np.random.randn(mbim.common_signal_available_knots.size))
        print(ll)

