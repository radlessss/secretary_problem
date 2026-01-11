import numpy as np
import random
import os

# Приховуємо технічні логи TensorFlow для чистого виводу
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from secretary_package.agent import agent_learner  # Ваш оновлений клас CooperativeAgentLearner
from secretary_package.utilfunctions import scale_state, single_shape_adaptor
from secretary_package.environment import TwoSideSecretaryEnv # Ваше нове середовище
# Припускаємо, що rl_utils адаптований або використовуємо стандартні списки для батчів

# 1. Ініціалізація кооперативного середовища
N_CANDIDATES = 100
env = TwoSideSecretaryEnv(N=N_CANDIDATES)

# 2. Параметри агента та навчання
nr_features = env.observation_space.shape[0]
nr_actions = env.action_space.n
max_replay_buffer_size = 50_000

# Створюємо агента (gamma=1.0 для кооперації)
agent_ler = agent_learner(nr_features=nr_features, nr_actions=nr_actions, gamma=1.0, learning_rate=0.0005)

# Налаштування навчання
U = 100 # Кількість оновлень Target-мережі
N_EPISODES = 5 # Епізодів між оновленнями Q
K_EPOCHS = 2   # Епох навчання на батчі

# Буфер пам'яті (використовуємо ваш клас Histories або список)
# Для диплому важливо: тут зберігаємо досвід ПАРИ
replay_buffer = [] 

for u in range(U):
    print(f"\n--- Update Target Q Round: {u}/{U} ---")
    
    # Епсилон-жадібна стратегія (зменшуємо дослідження з часом)
    epsilon = max(0.01, 0.2 - (0.2 - 0.01) * (u / U))
    print(f"Epsilon: {epsilon:.2%}")

    for n in range(N_EPISODES):
        # Отримуємо початкові стани для ОБОХ
        obs_man, obs_woman = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated:
            # Агент приймає рішення за чоловіка
            action_man = agent_ler.get_action(obs_man, env, epsilon=epsilon)
            
            # ТОЙ САМИЙ агент приймає рішення за жінку
            action_woman = agent_ler.get_action(obs_woman, env, epsilon=epsilon)

            # Крок у кооперативному середовищі
            (next_obs_man, next_obs_woman), reward, terminated, info = env.step(action_man, action_woman)

            # Зберігаємо досвід (спрощено для логіки навчання)
            # В кооперативній задачі ми вчимо стратегію "згоди"
            # Тому зберігаємо як дві окремі події, але з однаковою нагородою
            # Це і є "binding agreement" (обов'язкова угода)
            
            # Подія для чоловічої перспективи
            replay_buffer.append({
                'state': scale_state(obs_man, env),
                'action_idx': action_man,
                'reward': reward,
                'next_state': scale_state(next_obs_man, env),
                'done': terminated
            })
            
            # Подія для жіночої перспективи
            replay_buffer.append({
                'state': scale_state(obs_woman, env),
                'action_idx': action_woman,
                'reward': reward,
                'next_state': scale_state(next_obs_woman, env),
                'done': terminated
            })

            # Оновлюємо стани
            obs_man, obs_woman = next_obs_man, next_obs_woman
            steps += 1
            
            # Обмежуємо розмір буфера
            if len(replay_buffer) > max_replay_buffer_size:
                replay_buffer.pop(0)

        # Навчання на батчі після кожного епізоду
        if len(replay_buffer) > 64:
            for _ in range(K_EPOCHS):
                batch_indices = np.random.choice(len(replay_buffer), 64)
                batch = [replay_buffer[i] for i in batch_indices]
                # Створюємо об'єкти, які очікує ваш метод learn
                class Event:
                    def __init__(self, d):
                        self.scaled_state = d['state']
                        self.scaled_state_prime = d['next_state']
                        self.reward = d['reward']
                        self.done = d['done']
                        self.action_idx = d['action_idx']
                
                event_batch = [Event(d) for d in batch]
                agent_ler.learn(event_batch)

    # Оновлення Target-мережі (копіюємо ваги)
    agent_ler.update_Q_t_to_Q()
    
    # Вивід проміжного результату
    if u % 10 == 0:
        print(f"Update finished. Buffer size: {len(replay_buffer)}")

# saving the last agent
# --- Збереження результатів після завершення циклу навчання ---

# Шлях до файлу моделі (обов'язково додаємо .keras)
save_path = './training-results/Q-target/trained-agents/last-agent.keras'

# Перевіряємо, чи існує директорія, і створюємо її, якщо ні
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print("\n--- Навчання завершено! ---")
print(f"Збереження навченої моделі у: {save_path}")

# Зберігаємо Target-мережу, оскільки вона є результатом стабільної конвергенції
agent_ler.Q_t.save(save_path)

print("Модель успішно збережена на диск.")