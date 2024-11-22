# test_distributions.py

import unittest
from proppy.distributions import generate_bernoulli, generate_normal

class TestDistributions(unittest.TestCase):
    def test_generate_bernoulli(self):
        samples = generate_bernoulli(p=0.5, size=10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(sample in [0, 1] for sample in samples))
    
    def test_generate_normal(self):
        samples = generate_normal(mean=0, std=1, size=10)
        self.assertEqual(len(samples), 10)
