import unittest
from secretary_package.environment import SecretaryEnv
from secretary_package.utilfunctions import Averager
from secretary_package.utilfunctions import UniformDistributor, NormalDistributor, LogNormalDistributor

class TestCooperativeTwoSideSecretaryEnv(unittest.TestCase):
    def test_integration_with_averager(self):
        # 1. Створюємо реальні об'єкти
        avg = Averager()
        env = SecretaryEnv(num_sides=2, N=10, reward_func=avg, distributor=UniformDistributor(0, 1))
        obs = env.reset()
        self.assertAlmostEqual(obs[0][1], 0.0)  # Initial max_obs for agent 0
        self.assertAlmostEqual(obs[1][1], 0.0)  # Initial max_obs for agent 1

        self.assertAlmostEqual(obs[0][0], 0.1)
        self.assertAlmostEqual(obs[1][0], 0.1)

        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        
        # When action=0 (Continue), reward should be 0 and done should be False
        obs, reward, done, info = env.step([0, 0])
        
        # 2. Перевірки
        self.assertFalse(done)
        self.assertEqual(reward, 0)  # No reward when continuing
        self.assertEqual(avg.count, 0)  # Averager not used when continuing


        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ КРОКУ 1 (ACTION [0, 0]):")
        print(f"REWARD: {reward}")
        print(f"DONE:   {done}")
        print(f"INFO:   {info}")
        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        print("="*50 + "\n")
        # Now test with action=1 (Halt/Marriage)
        obs, reward, done, info = env.step([1, 1])

        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ КРОКУ 2 (ACTION [1, 1]):")
        print(f"REWARD: {reward}")
        print(f"DONE:   {done}")
        print(f"INFO:   {info}")
        # obs — це кортеж з двох масивів (obs_man, obs_woman)
        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        print("="*50)
        expected_reward = (env.current_qualities[0] + env.current_qualities[1]) / 2

        print(env.observations)
        self.assertTrue(done)
        self.assertAlmostEqual(reward, expected_reward, places=3)
        self.assertEqual(avg.count, 2)  # Both scores added

    def test_last_step(self):
        avg = Averager()
        env = SecretaryEnv(num_sides=1, N=3, reward_func=avg, distributor=UniformDistributor(0, 1))
        obs = env.reset()

        obs, reward, done, info = env.step([0])
        obs, reward, done, info = env.step([0])
        obs, reward, done, info = env.step([0])

        self.assertTrue(done)
   


    # def test_reset_clears_reward(self):
    #     avg = Averager()
    #     env = CooperativeTwoSideSecretaryEnv(N=10, reward_func=avg)
        
    #     # Робимо крок, щоб Averager наповнився даними
    #     env.step(1) 
    #     self.assertEqual(avg.count, 2)
        
    #     # Викликаємо reset і перевіряємо, чи обнулився Averager
    #     env.reset()
    #     self.assertEqual(avg.count, 0, "Reset має очищати Averager")

if __name__ == '__main__':
    unittest.main()