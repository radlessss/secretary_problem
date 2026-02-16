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


from secretary_package import SecretaryEnv
from secretary_package.threshold_agent import FixedThresholdStrategyAgent
from secretary_package.utilfunctions import Averager
import numpy as np

"""
    Функція, яка реалізує логіку симуляції для однінєї сторони використовуючи об'єкт оточення і агента.
    
    Ця функція реалізує класичну задачу секретаря для однієї сторони, 
    яка намагається оптимізувати свій вибір в залежності від переданого об'єкта агента. 

    Аргументи:
        environment: Ініціалізоване середовище.
        agent: Об'єкт агента, що реалізує стратегію прийняття рішень.
        episodes: Кількість незалежних епізодів симуляцій, які будуть проведені.

    Повертає:
        episodes_info: Список об'ктів з метриками отриманими під час проведення симуляції.
"""
def run_one_side_simulation(environment, agent, episodes=1000):

    episodes_info = []

    for _ in range(episodes):
        obs = environment.reset()
        agent.reset()
        terminated = False

        while not terminated:
            action = agent.make_decision(obs)
            obs, terminated, info = environment.step(action)

        episodes_info.append(info)

    return episodes_info




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
    episodes_info = []

    for _ in range(episodes):
        obs = environment.reset()
        agent1.reset()
        agent2.reset()
        terminated = False

        while not terminated:
            actions = [agent1.make_decision(obs[0]), agent2.make_decision(obs[1])]
            environment.render(mode='text')

            obs, terminated, info = environment.step(actions)

        episodes_info.append(info)

    return episodes_info




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
    episodes_info = []

    for _ in range(episodes):
        obs = environment.reset()
        agent.reset()
        terminated = False

        while not terminated:
            action = agent.make_decision(obs)
            environment.render(mode='text')

            obs, terminated, info = environment.step(action)

        episodes_info.append(info)

    return episodes_info



def evaluate_one_side_thresholds_scores(thresholds, distributor, episodes, N):
    avg_scores = []
    avg_steps = []
    avg_ranks = []
    avg_successes = []

    for th in thresholds:
        env = SecretaryEnv(num_sides=1, N=N, reward_func=Averager(), distributor=distributor)
        agent = FixedThresholdStrategyAgent(threshold=th)
        info = run_one_side_simulation(environment=env, agent=agent, episodes=episodes)

        # score/reward 
        # Беремо "reward" з кожного епізоду — це якість кандидата, якого агент обрав  
        scores = [i['reward'] for i in info] 
        # Усереднюємо по всіх епізодах → отримуємо середню якість обраного кандидата для цього threshold 
        avg_scores.append(np.mean(scores))   

        # success rate ranks
        # Беремо ранг обраного кандидата (1 = найкращий, 2 = другий і т.д.) з кожного епізоду
        ranks = [i['ranks'][0] for i in info] 
        # Конвертуємо ранги в успіх: 1, якщо обраний кандидат був найкращим, і 0, якщо ні
        successes = [1 if r == 1 else 0 for r in ranks]
        
        # Усереднюємо успіхи - ймовірність того, що агент вибирає найкращого кандидата для цього threshold
        avg_successes.append(np.mean(successes))
        # Усереднюємо ранги - середній ранг обраного кандидата для цього threshold
        avg_ranks.append(np.mean(ranks))

        # steps
        # Беремо крок (step), на якому агент зупинився в кожному епізоді
        steps = [i['step'] for i in info]
        # Усереднюємо кроки - середній крок вибору кандидата для цього threshold
        avg_steps.append(np.mean(steps))

    return avg_scores, avg_steps, avg_ranks, avg_successes
