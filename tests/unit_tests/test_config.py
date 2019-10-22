"""
Unit tests for methods in the configuration module.
"""

import unittest
from cryptonalysis.config import load_cryptonalysis_config_grid, load_cryptonalysis_config
from cryptonalysis.utils.misc_utils import all_elements_unique

RES_DIR = 'tests/resources'
CONFIG_TEST = 'config_test.conf'
CONFIG_GRID_TEST = 'config_grid_test.conf'


class TestConfig(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConfig, self).__init__(*args, **kwargs)
        self.config = load_cryptonalysis_config(RES_DIR, CONFIG_TEST)
        self.config_grid = load_cryptonalysis_config_grid(RES_DIR, CONFIG_GRID_TEST)

    def test_config(self):
        pc = self.config.preprocessing
        tc = self.config.training
        dlc = self.config.deep_learning

        # Test single-line and csv string representations
        pc_s = pc.to_single_line_str()
        self.assertEqual(pc_s, '2018-01-31TrueFalseBiffPredictor5111100CloseFalse2018-01-0110')
        pc_csv = pc.to_csv_str()
        self.assertEqual(pc_csv[0], u'end_date,normalize,normalize_by_row,predictor_cls,'
                                    u'predictor_params.daily_allowance,predictor_params.lookahead_days,'
                                    u'predictor_params.prob_buy,predictor_params.prob_sell,'
                                    u'predictor_params.starting_investment,price_column,standardize,start_date,'
                                    u'window_size')
        self.assertEqual(pc_csv[1], '2018-01-31,True,False,BiffPredictor,5,1,1,1,100,Close,False,2018-01-01,10')

        tc_s = tc.to_single_line_str()
        self.assertEqual(tc_s, '40.5True0.7')
        tc_csv = tc.to_csv_str()
        self.assertEqual(tc_csv[0], u'cv_folds,dev_size,shuffle_data,training_size')
        self.assertEqual(tc_csv[1], '4,0.5,True,0.7')

        dlc_s = dlc.to_single_line_str()
        self.assertEqual(dlc_s, '30')
        dlc_csv = dlc.to_csv_str()
        self.assertEqual(dlc_csv[0], u'neurons')
        self.assertEqual(dlc_csv[1], '30')

    def test_config_grid(self):
        cg = self.config_grid

        # Check grid sizes
        self.assertEqual(len(cg.cryptos), 4)
        self.assertEqual(len(cg.preprocessing_config_grid), 48)
        self.assertEqual(len(cg.training_config_grid), 1)
        self.assertEqual(len(cg.deep_learning_config_grid), 5)

        # Check all configs are different
        self.assertTrue(all_elements_unique(cg.preprocessing_config_grid))
        self.assertTrue(all_elements_unique(cg.training_config_grid))
        self.assertTrue(all_elements_unique(cg.deep_learning_config_grid))
