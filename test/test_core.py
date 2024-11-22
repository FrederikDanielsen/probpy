# test_core.py

import unittest
from probpy.core import main_function

class TestCore(unittest.TestCase):
    def test_main_function(self):
        # This test ensures the main function runs without errors.
        try:
            main_function()
        except Exception as e:
            self.fail(f"main_function raised an exception: {e}")
