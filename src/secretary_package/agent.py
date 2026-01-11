import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from secretary_package.utilfunctions import scale_state
import os


class agent_learner:
    def __init__(self, nr_features, nr_actions, gamma=1.0, learning_rate=0.0001):
        """
        Для кооперативної задачі встановлюємо gamma=1.0, 
        оскільки ми однаково цінуємо якість шлюбу в будь-який момент часу.
        """
        # Спільна Q-мережа (одна на обох)
        initializer = tf.keras.initializers.GlorotNormal()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        def build_network(name):
            inputs = keras.layers.Input(shape=(nr_features,), name=f'state_{name}')
            x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(inputs)
            x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(x)
            # Використовуємо linear для виходу, щоб мережа могла точно апроксимувати ранг
            outputs = layers.Dense(nr_actions, activation='linear', name=f'Q_{name}')(x)
            return keras.Model(inputs=inputs, outputs=outputs)

        self.Q = build_network('active')
        self.Q.compile(optimizer=optimizer, loss='mse')

        self.Q_t = build_network('target')
        self.update_Q_t_to_Q()

        self.gamma = gamma

    def prepare_learning_materials(self, events):
        """
        Обробка спільних подій кооперативної гри.
        У 'events' тепер зберігаються результати взаємодії ПАРИ.
        """
        nr_samples = len(events)
        
        # Витягуємо стани, нагороди та завершення
        s = np.array([x.scaled_state for x in events])
        s_primes = np.array([x.scaled_state_prime for x in events])
        r = np.array([x.reward for x in events]).reshape(-1, 1)
        done = np.array([x.done for x in events]).reshape(-1, 1)
        actions_taken = np.array([x.action_idx for x in events]) # Індекс дії (0 або 1)

        # 1. Передбачаємо значення для S' (Double DQN logic)
        # Використовуємо активну мережу для вибору найкращої дії в S'
        Q_s_prime_active = self.Q.predict(s_primes)
        best_actions_s_prime = np.argmax(Q_s_prime_active, axis=1)

        # Використовуємо Target-мережу для оцінки вартості цієї дії
        Q_s_prime_target = self.Q_t.predict(s_primes)
        
        # Витягуємо лише значення для найкращих дій
        rows = np.arange(nr_samples)
        future_values = Q_s_prime_target[rows, best_actions_s_prime].reshape(-1, 1)

        # 2. Розраховуємо цільове значення (Target Y) за формулою Беллмана
        # Q_target = r + gamma * Q_target(s', argmax Q_active(s'))
        rhs = r + self.gamma * future_values * (1 - done)

        # 3. Оновлюємо лише значення для виконаної дії
        Q_s_values = self.Q.predict(s)
        for i in range(nr_samples):
            Q_s_values[i, actions_taken[i]] = rhs[i]

        return s, Q_s_values

    def learn(self, events):
        X, y = self.prepare_learning_materials(events)
        self.Q.fit(X, y, epochs=1, verbose=0)

    def update_Q_t_to_Q(self):
        self.Q_t.set_weights(self.Q.get_weights())

    def get_action(self, state, env, epsilon):
        """
        Повертає дію (0 або 1). В кооперативній грі цей метод 
        викликається окремо для чоловіка та жінки.
        """
        scaled_state = np.array(scale_state(state, env)).reshape(1, -1)
        
        if np.random.rand() < epsilon:
            return np.random.randint(env.action_space.n)
        
        q_values = self.Q_t.predict(scaled_state)[0]
        return np.argmax(q_values)

print("TensorFlow завантажено успішно!")
print("Пристрій для обчислень:", tf.test.is_gpu_available() and "GPU" or "CPU")
