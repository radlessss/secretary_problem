import unittest
from secretary_package.utilfunctions import Multiplier, Adder, Averager

class TestUtilFunctions(unittest.TestCase):
    def test_multiplier(self):
        multiplier = Multiplier()
        multiplier.add_score(2)
        multiplier.add_score(3)
        self.assertEqual(multiplier.get_result(), 6)
        multiplier.reset()
        self.assertEqual(multiplier.get_result(), 1.0)

    def test_adder(self):
        adder = Adder()
        adder.add_score(2)
        adder.add_score(3)
        self.assertEqual(adder.get_result(), 5)
        adder.reset()
        self.assertEqual(adder.get_result(), 0.0)

    def test_averager(self):
        averager = Averager()
        averager.add_score(2)
        averager.add_score(4)
        self.assertEqual(averager.get_result(), 3.0)
        averager.reset()
        self.assertEqual(averager.get_result(), 0.0)

if __name__ == '__main__':
    unittest.main()
