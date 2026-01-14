# import numpy as np
# import tensorflow as tf
# import os
# from secretary_package.agent import agent_learner
# from secretary_package.secretary import CooperativeSecretaryEnv # Ваше нове середовище
# from secretary_package.utilfunctions import scale_state

# # Приховуємо логи TensorFlow для чистоти
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # 1. Ініціалізація середовища
# N_CANDIDATES = 100
# env = CooperativeSecretaryEnv(N=N_CANDIDATES)

# # 2. Створення та завантаження навченого агента
# nr_features = env.observation_space.shape[0]
# nr_actions = env.action_space.n
# agent_ler = agent_learner(nr_features=nr_features, nr_actions=nr_actions)

# # Завантажуємо ваги (мізки), які ви зберегли після learn.py
# model_path = './training-results/Q-target/trained-agents/last-agent.keras'
# agent_ler.Q_t = tf.keras.models.load_model(model_path)
# # Має бути (згідно з вашим класом в agent.py):
# agent_ler.update_Q_t_to_Q()

# # 3. Налаштування швидкого тесту
# # 1000 тестів достатньо для статистичної значущості і це займе лише кілька секунд
# nr_tests = 1000 
# rewards_lst = []
# steps_lst = []

# print(f"Запуск тестування на {nr_tests} епізодах...")

# for i in range(nr_tests):
#     obs_m, obs_w = env.reset()
#     terminated = False
    
#     while not terminated:
#         # Використовуємо epsilon=0, бо ми перевіряємо вже навчену стратегію
#         action_m = agent_ler.get_action(obs_m, env, epsilon=0)
#         action_w = agent_ler.get_action(obs_w, env, epsilon=0)
        
#         # Крок у кооперативному середовищі
#         (obs_m, obs_w), reward, terminated, info = env.step(action_man=action_m, action_woman=action_w)
        
#     rewards_lst.append(reward)
#     steps_lst.append(env.time)

# # 4. Розрахунок ключових показників для диплома
# avg_reward = np.mean(rewards_lst)
# # Очікуваний ранг RN при N->inf пов'язаний з балом (Score) як E[Rank] = N * (1 - Score)
# avg_rank = N_CANDIDATES * (1 - avg_reward)
# # Константа з Conjecture 2: RN / sqrt(N)
# calculated_constant = avg_rank / np.sqrt(N_CANDIDATES)

# print("-" * 30)
# print(f"РЕЗУЛЬТАТИ ТЕСТУ (N={N_CANDIDATES}):")
# print(f"Середній бал (якість): {avg_reward:.4f}")
# print(f"Середній ранг (RN): {avg_rank:.2f}")
# print(f"Отримана константа: {calculated_constant:.4f} (Очікувалось ~0.92)")
# print("-" * 30)

# # 5. Збереження результатів для побудови графіків
# perfdir = './performance-and-animations/'
# if not os.path.exists(perfdir): os.makedirs(perfdir)

# np.savetxt(perfdir + 'cooperative_rewards.dat', rewards_lst)
# np.savetxt(perfdir + 'cooperative_steps.dat', steps_lst)
# print(f"Дані збережені в {perfdir}")



def run_simulation(environment, agent1, agent2, episodes=1000):
    rewards = []
    steps = []

    for _ in range(episodes):
        obs_m, obs_w = environment.reset()
        agent1.reset()
        agent2.reset()
        terminated = False

        while not terminated:
            action_m = agent1.get_action(obs_m, environment, epsilon=0)
            action_w = agent2.get_action(obs_w, environment, epsilon=0)
            environment.render(mode='text')

            (obs_m, obs_w), reward, terminated, info = environment.step(
                action_man=action_m,
                action_woman=action_w
            )

        rewards.append(reward)
        steps.append(environment.time)

    return rewards, steps
