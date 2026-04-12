import unittest
import numpy as np
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

        self.assertAlmostEqual(obs[0][0], 1)
        self.assertAlmostEqual(obs[1][0], 1)

        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        
        # When action=0 (Continue), reward should be 0 and done should be False
        obs, done, info = env.step([0, 0])
        
        # 2. Перевірки
        self.assertFalse(done)
        self.assertEqual(avg.count, 0)  # Averager not used when continuing


        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ КРОКУ 1 (ACTION [0, 0]):")
#        print(f"REWARD: {reward}")
        print(f"DONE:   {done}")
        print(f"INFO:   {info}")
        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        print("="*50 + "\n")
        # Now test with action=1 (Halt/Marriage)
        obs, done, info = env.step([1, 1])

        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ КРОКУ 2 (ACTION [1, 1]):")
#        print(f"REWARD: {reward}")
        print(f"DONE:   {done}")
        print(f"INFO:   {info}")
        # obs — це кортеж з двох масивів (obs_man, obs_woman)
        print(f"OBS Агента 0: {obs[0]}")
        print(f"OBS Агента 1: {obs[1]}")
        print("="*50)
        expected_reward = (env.current_qualities[0] + env.current_qualities[1]) / 2

        print(env.observations)
        self.assertTrue(done)
#        self.assertAlmostEqual(reward, expected_reward, places=3)
        self.assertEqual(avg.count, 2)  # Both scores added

    # def test_last_step(self):
    #     avg = Averager()
    #     env = SecretaryEnv(num_sides=1, N=3, reward_func=avg, distributor=UniformDistributor(0, 1))
    #     obs = env.reset()
    #     print(f"Початковий стан (Reset). OBS: {obs[0]}")

    #     for step_num in range(1, 4):
    #         # Передаємо action [0] (продовжувати пошук)
    #         obs, done, info = env.step([0])
            
    #         print("\n" + "-"*40)
    #         print(f"КРОК {step_num} (ACTION [0]):")
    #         print(f"DONE:   {done}")
    #         print(f"INFO:   {info}")
    #         print(f"OBS:    {obs[0]}")

    #         # На останньому кроці (коли кандидати закінчилися) done має стати True
    #         if step_num == 3:
    #             print("Досягнуто ліміту N=3. Перевірка завершення...")
    #             self.assertTrue(done)
    #         else:
    #             self.assertFalse(done)
        
    #     print("-"*40 + "\n")

    #     self.assertTrue(done)

    def test_last_step(self):
        print(f"\n{'#'*30}\nSTART: test_last_step\n{'#'*30}")
        avg = Averager()
        N_limit = 3
        env = SecretaryEnv(num_sides=1, N=N_limit, reward_func=avg, distributor=UniformDistributor(0, 1))
        obs = env.reset()
        
        # Початкова якість першого кандидата (зберігаємо для звірки)
        initial_quality = env.current_qualities[0]

        for step_num in range(1, N_limit + 1):
            obs, done, info = env.step([0])
            
            print(f"\nКРОК {step_num} (ACTION [0]):")
            print(f"DONE: {done} | INFO: {info}")
            
            # --- ПЕРЕВІРКА 1: Формат OBS (має бути завжди списком/масивом) ---
            self.assertIsInstance(obs, (list, np.ndarray), f"На кроці {step_num} OBS має бути масивом!")
            # self.assertEqual(len(obs[0]), 3, "Спостереження агента має містити 3 елементи")

            # --- ПЕРЕВІРКА 2: Логіка завершення ---
            if step_num < N_limit:
                self.assertFalse(done, f"На кроці {step_num} гра не мала завершитися")
                self.assertEqual(info['msg'], 'Next candidate')
            else:
                # Це останній крок (Last Resort)
                print("!!! Досягнуто ліміту. Перевірка логіки завершення...")
                self.assertTrue(done, "На останньому кроці (N=3) done має бути True")
                self.assertEqual(info['msg'], 'Last resort', "Має бути повідомлення 'Last resort'")
                
                # --- ПЕРЕВІРКА 3: Винагорода на останньому кроці ---
                # Оскільки ми тиснули "0" (пропустити), на останньому кроці 
                # нам мають примусово віддати останнього кандидата.
                last_candidate_quality = env.current_qualities[0]
                self.assertIn('reward', info, "В info має бути ключ 'reward'")
                self.assertAlmostEqual(info['reward'], last_candidate_quality, places=5, 
                                     msg="Винагорода має дорівнювати якості останнього кандидата")

        print("\n" + "#"*30 + "\nTEST PASSED\n" + "#"*30)
   


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