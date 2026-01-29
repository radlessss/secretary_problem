import unittest
from secretary_package.threshold_agent import CooperativeTwoSideThresholdAgent
from secretary_package.utilfunctions import Multiplier, Adder, Averager

class TestThresholdAgent(unittest.TestCase):

    def test_get_action_below_threshold(self):
        agent = CooperativeTwoSideThresholdAgent(threshold=0.37, score_calculation_func=Multiplier())
        obs1 = (0.2, None, 0.5)
        obs2 = (0.2, None, 0.2)
        action = agent.make_decision(obs1, obs2, environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.1)  # 0.5 * 0.2  

    def test_cooperative_agent_positive_descision(self):
        obs = [0.1, 0.2, 0.3, 0.4, 0.5]
        agent = CooperativeTwoSideThresholdAgent(threshold=0.2, score_calculation_func=Adder())

        action = agent.make_decision([0.1, 0, 0.1], [0.1, 0, 0.1], environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.2)  # 0.1

        action = agent.make_decision([0.2, 0, 0.2], [0.2, 0, 0.2], environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.4)  

        action = agent.make_decision([0.3, 0, 0.3], [0.3, 0, 0.3], environment=None)
        self.assertEqual(action, 1)
        self.assertEqual(agent.max_obs, 0.6)  

    
    def test_cooperative_agent_negative_descision(self):
        obs = [0.1, 0.2, 0.3, 0.4, 0.5]
        agent = CooperativeTwoSideThresholdAgent(threshold=0.2, score_calculation_func=Adder())

        action = agent.make_decision([0.1, 0, 0.3], [0.1, 0, 0.3], environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.6) 

        action = agent.make_decision([0.2, 0, 0.2], [0.2, 0, 0.2], environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.6)  

        action = agent.make_decision([0.3, 0, 0.1], [0.3, 0, 0.1], environment=None)
        self.assertEqual(action, 0)
        self.assertEqual(agent.max_obs, 0.6)  


if __name__ == '__main__':
    unittest.main()