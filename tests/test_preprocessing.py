import unittest
from preprocessing import scale_age

class TestPreprocessing(unittest.TestCase):
    def test_scale_age(self):
        self.assertEqual(scale_age(90), -210)  # Test that ages above 89 are handled correctly
        self.assertEqual(scale_age(50), 50)    # Test that ages below 89 remain unchanged

if __name__ == '__main__':
    unittest.main()
