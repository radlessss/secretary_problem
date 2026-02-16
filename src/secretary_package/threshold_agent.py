from secretary_package.utilfunctions import Multiplier, Adder, Averager
"""
    Реалізація базового агента для вирішення задачі про секретаря 
    з використанням порогової стратегії.

    Логіка роботи:
    1. Етап спостереження: 
       Поки поточний крок (step) менший або дорівнює пороговому значенню threshold, 
       агент лише спостерігає за кандидатами, оновлює значення найкращої якості max_obs, 
       яку він бачив, і відхиляє всіх кандидатів повертає [0].
    
    2. Етап відбору: 
       Після проходження порогу агент починає шукати кандидата. 
       Він обирає (повертає [1]) першого ж кандидата, чия якість вища або дорівнює 
       найкращій якості, зафіксованій на етапі спостереження.

    Атрибути:
        threshold: Поріг, до якого агент лише збирає інформацію.
        max_obs: Максимальна якість кандидата, знайдена на даний момент.
"""
class FixedThresholdStrategyAgent:
    def __init__(self, threshold):
        if threshold <= 0 or threshold >= 1 or threshold is None:
            raise ValueError("Threshold must be in (0, 1) range and not None.")
        
        self.threshold = threshold
        self.max_obs = 0
    
    def reset(self):
        self.max_obs = 0

#    def make_decision(self, obs):
#        if len(obs) != 1:
#            raise ValueError("Observation must contain state from one side.")
#        step, _, current_quality = obs[0]
#        if self.threshold >= step:
#            self.max_obs = max(self.max_obs, current_quality)
#           return [0] 
#      elif current_quality >= self.max_obs:
#            return [1]
#        else:
#            return [0]

    def make_decision(self, obs):
        if len(obs) != 1:
            raise ValueError("Observation must contain state from one side.")

        step, _, current_quality = obs[0]

        # skip phase
        if step < self.threshold:
            self.max_obs = max(self.max_obs, current_quality)
            return [0]

        # accept phase
        if current_quality > self.max_obs:
            self.max_obs = current_quality
            return [1]

        self.max_obs = max(self.max_obs, current_quality)
        return [0]


# to do add threshold agent implementation


class FixedThresholdStrategyAgentProbne:
    def __init__(self, threshold):
        if threshold <= 0 or threshold >= 1 or threshold is None:
            raise ValueError("Threshold must be in (0, 1) range and not None.")
        
        self.threshold = threshold
        self.max_obs = 0
    
    def reset(self):
        self.max_obs = 0

    def make_decision(self, obs):
        # obs — це numpy.array([step, max_score_so_far, current_quality])
        # Перевіряємо, чи це дійсно масив з 3 елементами (якщо хочете залишити перевірку)
        if len(obs) != 3:
            raise ValueError(f"Observation must contain 3 elements, got {len(obs)}.")

        # Пряме розпакування масиву
        step, _, current_quality = obs

        # 1. Фаза спостереження (skip phase)
        # Оскільки step у вас — це (time+1)/N, порівнюємо з числовим порогом
        if step < self.threshold:
            self.max_obs = max(self.max_obs, current_quality)
            return 0  # Повертаємо число, а не список [0]

        # 2. Фаза вибору (accept phase)
        if current_quality > self.max_obs:
            return 1  # Погоджуємось
        
        # Оновлюємо рекорд, якщо не погодились
        self.max_obs = max(self.max_obs, current_quality)
        return 0


"""
    Реалізація кооперативного агента, який приймає рішення на основі об'єднаних даних 
    з двох сторін.

    Логіка роботи:
    1. Агрегація оцінок: 
       На кожному кроці агент отримує спостереження з двох джерел (obs[0] та obs[1]).
       Використовує передану функцію (score_calculation_func), щоб об'єднати ці дві оцінки 
       в одну загальну метрику якості (current_quality).
    
    2. Етап спостереження: 
       Якщо поточний крок не перевищив поріг threshold, агент оновлює глобальний максимум 
       спільної оцінки max_obs і пропускає хід.
    
    3. Етап відбору: 
       Якщо поріг пройдено, агент обирає пару, сумарна якість якої перевищує або дорівнює 
       найкращій якості, яку було зафіксовано раніше.

    Атрибути:
        threshold: Нормалізований поріг (від 0 до 1), що визначає тривалість фази навчання.
        score_calculation_func: Об'єкт (наприклад, Adder, Multiplier), що комбінує оцінки.
"""


class CooperativeTwoSideThresholdAgent:
    def __init__(self, threshold, score_calculation_func):
        if threshold <= 0 or threshold >= 1 or threshold is None:
            raise ValueError("Threshold must be in (0, 1) range and not None.")
        
        self.threshold = threshold

        if score_calculation_func is None:
            raise ValueError("score_calculation_func is None.")
        
        self.score_calculation_func = score_calculation_func
        self.max_obs = 0
    
    def reset(self):
        self.max_obs = 0
        self.score_calculation_func.reset()

    def make_decision(self, obs):
        self.score_calculation_func.reset()
        if len(obs) != 2:
            raise ValueError("Observation must contain states from both sides.")

        step, _, current_quality1 = obs[0]
        step, _, current_quality2 = obs[1]
        
        self.score_calculation_func.add_score(current_quality1)
        self.score_calculation_func.add_score(current_quality2)
        current_quality = self.score_calculation_func.get_result()
        
        # Always update max_obs to track the best quality seen
        self.max_obs = max(self.max_obs, current_quality)
    
        if self.threshold >= step:
            return [0] 
        elif current_quality >= self.max_obs:
            return [1]
        else:
            return [0]
        


