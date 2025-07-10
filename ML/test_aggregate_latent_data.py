import unittest
import numpy as np
from F21Stats import F21Stats  # Adjust the import based on your file structure

class TestAggregateLatentData(unittest.TestCase):

    def test_aggregate_basic(self):
        params = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        latent_features = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        num_rows = 2
        
        keys, means = F21Stats.aggregate_f21_data(params, latent_features, num_rows)
        
        expected_keys = np.array([[1.00,2.00], [1.00, 2.00], [3.00, 4.00]])
        expected_means = np.array([[1.5, 2.5], [3.0, 4.0],[4.0, 5.0]])
        
        np.testing.assert_array_equal(keys, expected_keys)
        np.testing.assert_array_equal(means, expected_means)

    def test_aggregate_with_different_keys(self):
        params = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
        latent_features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        num_rows = 2
        
        keys, means = F21Stats.aggregate_f21_data(params, latent_features, num_rows)
        
        expected_keys = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_means = np.array([[3.0, 4.0], [5.0, 6.0]])
        
        np.testing.assert_array_equal(keys, expected_keys)
        np.testing.assert_array_equal(means, expected_means)

    def test_aggregate_with_excess_rows(self):
        params = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        latent_features = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        num_rows = 2
        
        keys, means = F21Stats.aggregate_f21_data(params, latent_features, num_rows)
        
        expected_keys = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        expected_means = np.array([[1.5, 2.5], [3.5, 4.5], [5.0, 6.0]])
        
        np.testing.assert_array_equal(keys, expected_keys)
        np.testing.assert_array_equal(means, expected_means)

    def test_empty_input(self):
        params = np.array([])
        latent_features = np.array([])
        num_rows = 2
        
        keys, means = F21Stats.aggregate_f21_data(params, latent_features, num_rows)
        
        expected_keys = np.array([])
        expected_means = np.array([])
        
        np.testing.assert_array_equal(keys, expected_keys)
        np.testing.assert_array_equal(means, expected_means)

if __name__ == '__main__':
    unittest.main()