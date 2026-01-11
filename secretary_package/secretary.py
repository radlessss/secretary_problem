import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class CooperativeSecretaryEnv(gym.Env):
    '''
    Двостороннє кооперативне середовище задачі секретаря.
    Успіх (шлюб) можливий лише при взаємній згоді обох сторін.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, N=100):
        super(CooperativeSecretaryEnv, self).__init__()
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
        
        return self._get_observations()

    def _get_observations(self):
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

        observations = self._get_observations()
        return observations, reward, done, info

    def render(self, mode='human'):
        if mode == 'text':
            print(f"Раунд: {self.time}/{self.N} | Якість М: {self.current_man_quality:.2f} | Якість Ж: {self.current_woman_quality:.2f}")

    def close(self):
        print('Середовище закрито.')