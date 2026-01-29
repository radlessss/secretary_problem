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

"""
    Симуляція для одного агента у заданому середовищі.
    
    Ця функція підходить для класичної задачі секретаря або однієї сторони, 
    яка намагається оптимізувати свій вибір незалежно від інших (або якщо середовище 
    абстрагує другу сторону).

    Аргументи:
        environment: Ініціалізоване середовище (наприклад, OneSideSecretaryEnv).
        agent: Об'єкт агента, що має методи `reset()` та `make_decision(obs)`.
        episodes: Кількість незалежних епізодів (ігор) для запуску. За замовчуванням 1000.

    Повертає:
        rewards: Список отриманих винагород за кожен епізод.
        steps: Список кількості кроків (скільки кандидатів було переглянуто) для кожного епізоду.
"""
def run_one_side_simulation(environment, agent, episodes=1000):

    rewards = []
    steps = []

    for _ in range(episodes):
        obs = environment.reset()
        agent.reset()
        terminated = False

        while not terminated:
            action = agent.make_decision(obs)
            #environment.render(mode='text')

            obs, reward, terminated, info = environment.step(action)

        rewards.append(reward)
        steps.append(environment.time)

    return rewards, steps




"""
    Симуляція взаємодії ДВОХ незалежних агентів у спільному середовищі.
    
    Використовується для `TwoSideSecretaryEnv`, де є два окремі учасники (наприклад, Чоловік і Жінка),
    кожен з яких має власну стратегію і бачить тільки свою частину спостережень.

    Аргументи:
        environment (gym.Env): Двостороннє середовище.
        agent1 (object): Агент першої сторони (obs[0]).
        agent2 (object): Агент другої сторони (obs[1]).
        episodes (int): Кількість епізодів.

    Логіка:
        Середовище повертає кортеж спостережень (obs_man, obs_woman).
        Кожен агент отримує своє спостереження і генерує власну дію.
        Дії об'єднуються у список і передаються в середовище.

    Повертає:
        rewards (list), steps (list)
"""
def run_two_side_simulation(environment, agent1, agent2, episodes=1000):
    rewards = []
    steps = []

    for _ in range(episodes):
        obs = environment.reset()
        agent1.reset()
        agent2.reset()
        terminated = False

        while not terminated:
            actions = [agent1.make_decision(obs[0]), agent2.make_decision(obs[1])]
            environment.render(mode='text')

            obs, reward, terminated, info = environment.step(actions)

        rewards.append(reward)
        steps.append(environment.time)

    return rewards, steps




"""
    Симуляція для "Кооперативного" агента.
    
    У цьому випадку `agent` виступає як централізований контролер (наприклад, `CooperativeTwoSideThresholdAgent`),
    який бачить повну картину (обидва боки спостережень) і приймає узгоджене рішення 
    для системи в цілому.

    Аргументи:
        environment (gym.Env): Середовище (наприклад, SecretaryEnv з num_sides=2).
        agent (object): Кооперативний агент, метод `make_decision` якого очікує повний набір спостережень.
        episodes (int): Кількість епізодів.

    Логіка:
        Агент отримує `obs` (який містить дані всіх сторін) і повертає дію (або список дій),
        які ведуть до спільної мети (максимізації сумарної винагороди).

    Повертає:
        rewards (list), steps (list)
"""
def run_cooperative_two_side_simulation(environment, agent, episodes=1000):
    rewards = []
    steps = []

    for _ in range(episodes):
        obs = environment.reset()
        agent.reset()
        terminated = False

        while not terminated:
            action = agent.make_decision(obs)
            environment.render(mode='text')

            obs, reward, terminated, info = environment.step(action)

        rewards.append(reward)
        steps.append(environment.time)

    return rewards, steps


