import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from secretary_package.utilfunctions import Averager
from secretary_package.utilfunctions import UniformDistributor, NormalDistributor, LogNormalDistributor

# TO DO: impement ability to provide functional reward object for calculation of reward
# it should be implemented in the same way as score_calculation_func in CooperativeTwoSideThresholdAgent
# by default it should use Averager object
# learn about test mock objects for testing purposes and testing this class  and environment object in particular


"""
    Двостороннє кооперативне середовище для задачі про секретаря (Two-Sided Secretary Problem).

    Опис:
    Це середовище моделює ситуацію, де дві сторони (наприклад, Чоловік та Жінка, або Роботодавець та Кандидат)
    одночасно оцінюють одне одного. Рішення про "шлюб" (зупинку пошуку) приймається тільки тоді,
    коли ОБИДВІ сторони погоджуються (дія 1).

    Основна логіка:
    1. Ініціалізація: Задається кількість кандидатів N.
    2. Спостереження (Observation):
       Кожен агент бачить вектор з 3-х значень:
       - Нормалізований час (t/N).
       - Найкраща якість партнера, яку цей агент бачив раніше (Max Score).
       - Якість поточного партнера (Current Quality).
    3. Крок (Step):
       Приймаються дві дії (від чоловіка та жінки).
       - Якщо обидва = 1: Гра завершується, нараховується винагорода.
       - Якщо хоча б один = 0: Гра продовжується, переходимо до наступної пари кандидатів.
    4. Винагорода (Reward):
       На даний момент реалізована як середнє арифметичне якостей обох партнерів у вибраній парі.
       Якщо дійшли до кінця (N) без вибору, обирається остання пара ("Last resort").

    Атрибути:
        N: Загальна кількість кандидатів (довжина горизонту).
        action_space: 0 (Пропустити) або 1 (Обрати).
        observation_space: Вектор станів [час, рекорд, поточна_якість].
"""
class TwoSideSecretaryEnv(gym.Env):
    '''
    Двостороннє кооперативне середовище задачі секретаря.
    Успіх (шлюб) можливий лише при взаємній згоді обох сторін.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, N=100):
        super(TwoSideSecretaryEnv, self).__init__()
        self.N = N
        self.action_space = spaces.Discrete(2) # 0: Continue, 1: Halt
        
        # Стан (Observation) для одного агента: [час, рекорд_партнера, якість_поточного_партнера]
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.time = 1
        # Найкращі бали, які бачили сторони до цього моменту
        self.max_man_score = 0.0
        self.max_woman_score = 0.0
        
        # Генеруємо початкову пару
        self.current_man_quality = self.np_random.uniform(0, 1)   # Наскільки гарний чоловік
        self.current_woman_quality = self.np_random.uniform(0, 1) # Наскільки гарна жінка
        
        return self._generate_next_observations()

    def _generate_next_observations(self):
        '''Повертає стан окремо для чоловіка і для жінки'''
        # Чоловік бачить жінку (її якість та свій рекорд у пошуку жінок)
        obs_man = np.array([self.time/self.N, self.max_woman_score, self.current_woman_quality], dtype=np.float32)
        # Жінка бачить чоловіка
        obs_woman = np.array([self.time/self.N, self.max_man_score, self.current_man_quality], dtype=np.float32)
        return obs_man, obs_woman

    def step(self, action_man, action_woman):
        '''
        Отримує дії обох сторін.
        action_man: 1 (Halt/Пропозиція), 0 (Continue/Відмова)
        action_woman: 1 (Halt/Згода), 0 (Continue/Відмова)
        '''
        # Шлюб відбувається лише якщо обидва сказали 1
        marriage = (action_man == 1 and action_woman == 1)
        
        # Якщо шлюб відбувся або ми дійшли до останнього кандидата
        if marriage or self.time >= self.N:
            done = True
            # Нагорода в кооперативній грі — це середня якість партнерів
            # Або якість того, кого ви отримали в результаті
            reward = (self.current_man_quality + self.current_woman_quality) / 2
            info = {'msg': 'Marriage Success' if marriage else 'Last resort marriage'}
        else:
            # Гра продовжується
            done = False
            reward = 0
            
            # Оновлюємо рекорди
            self.max_man_score = max(self.max_man_score, self.current_man_quality)
            self.max_woman_score = max(self.max_woman_score, self.current_woman_quality)
            
            # Новий раунд, нова пара
            self.time += 1
            self.current_man_quality = self.np_random.uniform(0, 1)
            self.current_woman_quality = self.np_random.uniform(0, 1)
            info = {'msg': 'Next candidate'}
            # TO DO: provide enabitily to extract information about rank

        observations = self._generate_next_observations()
        return observations, reward, done, info

    def render(self, mode='human'):
        if mode == 'text':
            print(f"Раунд: {self.time}/{self.N} | Якість М: {self.current_man_quality:.2f} | Якість Ж: {self.current_woman_quality:.2f}")




#TO DO: implement CooperativeTwoSideSecretaryEnv class that will inherit from TwoSideSecretaryEnv
# override step method 
# class CooperativeTwoSideSecretaryEnv(TwoSideSecretaryEnv):
#     pass


"""
    Узагальнене середовище задачі секретаря для довільної кількості сторін (num_sides).

    Опис:
    На відміну від TwoSideSecretaryEnv, цей клас підтримує N сторін (наприклад, 3+ агентів, 
    які повинні дійти згоди). Також він використовує патерн Strategy для обчислення фінального 
    рахунку через `reward_func`.

    Основна логіка:
    1. Ініціалізація: Приймає кількість сторін та об'єкт функції винагороди (наприклад, Averager, Multiplier).
    2. Спостереження: Повертає список станів для кожного з `num_sides` агентів.
    3. Крок (Step):
       - Приймає список дій `actions`.
       - Шлюб (зупинка) відбувається ТІЛЬКИ якщо `all(actions) == 1` (усі погодилися).
       - При успіху або вичерпанні часу обчислюється винагорода за допомогою `reward_func`.
    
    Атрибути:
        num_sides: Кількість агентів/сторін.
        reward_func: Об'єкт, що має методи `add_score()` та `get_result()` (поліморфізм).
        current_qualities: Поточні якості кандидатів для кожної сторони.
"""
class SecretaryEnv(gym.Env):
    def __init__(self, num_sides, N, reward_func, distributor):
        super(SecretaryEnv, self).__init__()
        
        if num_sides < 1:
            raise ValueError("Number of sides must be at least 1.")
        self.num_sides = num_sides
        
        if N < 2:
            raise ValueError("N must be at least 2.")
        self.N = N
        
        if distributor is None:
            raise ValueError("Distributor must be added as parameter and not None.")
        self.distributor = distributor

        if reward_func is None:
            raise ValueError("Reward function must be added as parameter and not None.")
        self.reward_func = reward_func
        
        self.time = -1
        self.max_scores = [0.0 for _ in range(num_sides)]
        self.observations = self.generate_observations()
        self.current_qualities = list(self.observations[self.time])
        
        #self.action_space = spaces.Discrete(2) # 0: Continue, 1: Halt

        # Стан (Observation) для одного агента: [час, рекорд_партнера, якість_поточного_партнера]
        #self.observation_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float32)
        
    def generate_observations(self):
        observations = []
        for i in range(self.N):
            obs = []
            for j in range(self.num_sides):
                obs.append(self.distributor.sample())
            observations.append(tuple(obs))
        return observations
    
    def reset(self):
        self.time = -1
        self.reward_func.reset()
        # Найкращі бали, які бачили сторони до цього моменту
        self.max_scores = [0.0 for _ in range(self.num_sides)]
        self.observations = self.generate_observations()
        
        return self._generate_next_observations()

    def _generate_next_observations(self):
        '''Повертає стан окремо для чоловіка і для жінки'''
        # Генеруємо початкову пару
        
        self.time += 1

        if self.time >= self.N:
            raise Exception(f" Поточний крок вийшов за межі кроку симуляції time={self.time}, N={self.N}")
        
        self.current_qualities = list(self.observations[self.time])
        obs = []
        for i in range(0, self.num_sides):
            obs_i = np.array([(self.time+1)/self.N, self.max_scores[i], self.current_qualities[i]], dtype=np.float32)
            obs.append(obs_i)
        return obs
    

    def get_absolute_ranks(self):
        agents_current_agent_qualities_ranks = []

        def compute_rank(qualities, q):
            sorted_qualities = sorted(qualities, reverse=True)
            return sorted_qualities.index(q) + 1
        
        for agent_idx, current_agent_quality in enumerate(self.current_qualities):
            agent_values = [value [agent_idx] for value in self.observations]
            current_agent_quality_rank = compute_rank(agent_values, current_agent_quality)
            agents_current_agent_qualities_ranks.append(current_agent_quality_rank)

        return agents_current_agent_qualities_ranks


    def step(self, actions=[]): 

        # Шлюб відбувається лише за взаємної згоди
        marriage = all(action == 1 for action in actions)
        observations = None
        if marriage or self.time >= (self.N - 1):
            done = True
            
            self.reward_func.reset()

            for quality in self.current_qualities:
                self.reward_func.add_score(quality)

            reward = self.reward_func.get_result()
            observations = self.current_qualities

            info = {}
            info['msg'] = 'Marriage Success'  if marriage else 'Last resort'
            info['observations'] = self.current_qualities
            info['ranks'] = self.get_absolute_ranks()
            info['reward'] = self.reward_func.get_result()
            info['step'] = self.time + 1
         #   info['fraction'] = (self.time + 1) / self.N


        else:
            done = False
            reward = 0

            # Оновлюємо рекорди (те, що агенти бачили до цього)
            for i, quality in enumerate(self.current_qualities):
                self.max_scores[i] = max(self.max_scores[i], quality)

            info = {'msg': 'Next candidate'}
            observations = self._generate_next_observations()
 # to do: add in return "step = self.time"  - done, info
        return observations, done, info
    
    def render(self, mode='human'):
        if mode == 'text':
            print(f"Раунд: {self.time}/{self.N} | Якість М: {self.current_qualities[0]:.2f} | Якість Ж: {self.current_qualities[1]:.2f}")



# # він має приймати кількість сторін, кількість кроків, функцію розрахунку балу (reward_func)
# class SecretaryEnv(TwoSideSecretaryEnv):
#     def __init__(self, num_sides=2, N=100, reward_func=None):
#         self.num_sides = num_sides
#         self.N = N
#         self.reward_func = reward_func if reward_func is not None else Averager()
        
#         super(SecretaryEnv, self).__init__(N)

#         self.action_space = spaces.Discrete(2) # 0: Continue, 1: Halt
        
#         # Стан (Observation) для одного агента: [час, рекорд_партнера, якість_поточного_партнера]
#         self.observation_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float32)
        
#         self.seed()
#         self.reset()

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def reset(self):
#         self.time = 0
#         self.reward_func.reset()
#         # Найкращі бали, які бачили сторони до цього моменту
#         self.max_man_score = 0.0
#         self.max_woman_score = 0.0
        
#         return self._generate_next_observations()

#     def _generate_next_observations(self):
#         '''Повертає стан окремо для чоловіка і для жінки'''
#         # Генеруємо початкову пару
#         self.time += 1
#         self.current_man_quality = self.np_random.uniform(0, 1)   # Наскільки гарний чоловік
#         self.current_woman_quality = self.np_random.uniform(0, 1) # Наскільки гарна жінка
#         # Чоловік бачить жінку (її якість та свій рекорд у пошуку жінок)
#         obs_man = np.array([self.time/self.N, self.max_woman_score, self.current_woman_quality], dtype=np.float32)
#         # Жінка бачить чоловіка
#         obs_woman = np.array([self.time/self.N, self.max_man_score, self.current_man_quality], dtype=np.float32)
#         return obs_man, obs_woman

#     def step(self, action): 

#         # Шлюб відбувається лише за взаємної згоди
#         marriage = (action == 1)
        
#         if marriage or self.time >= self.N:
#             done = True

#             self.reward_func.add_score(self.current_man_quality)
#             self.reward_func.add_score(self.current_woman_quality)
#             reward = self.reward_func.get_result()
#             info = {'msg': 'Marriage Success' if marriage else 'Last resort'}
#             observations = (self.time, self.current_man_quality, self.current_woman_quality)
#         else:
#             done = False
#             reward = 0

#             # Оновлюємо рекорди (те, що агенти бачили до цього)
#             self.max_man_score = max(self.max_man_score, self.current_man_quality)
#             self.max_woman_score = max(self.max_woman_score, self.current_woman_quality)
#             info = {'msg': 'Next candidate'}
#             observations = self._generate_next_observations()
            
#         return observations, reward, done, info