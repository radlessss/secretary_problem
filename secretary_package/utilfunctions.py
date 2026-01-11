import numpy as np

def scale_state(state, env):
    '''
    Масштабування координат стану в діапазон від -1.0 до 1.0.
    Необхідно для коректної роботи активаційної функції tanh.
    '''
    infinity = 10
    low = env.observation_space.low
    high = env.observation_space.high

    # Обмеження нескінченних меж
    low = np.array([x if x > -infinity else -infinity for x in low])
    high = np.array([x if x < infinity else infinity for x in high])

    mean = 0.5 * (high + low)
    range_ = high - low
    
    scaled_state = 2.0 * (state - mean) / (range_ + 1e-7)
    return scaled_state

def single_shape_adaptor(state, nr_features):
    '''
    Перетворює стан з форми (nr_features,) у форму (1, nr_features).
    Це необхідно, тому що TensorFlow очікує вхідні дані у вигляді батчу.
    '''
    return np.reshape(np.array([state]), (1, nr_features))

def one_hot(chosen_act, nr_actions):
    '''Перетворює індекс дії в one-hot вектор'''
    tmp = np.zeros(nr_actions)
    tmp[chosen_act] = 1
    return tmp

def initializer(initial_state):
    '''Ініціалізує змінні на початку кожного епізоду'''
    state = np.array([initial_state])
    terminated = False
    steps = 0
    return state, terminated, steps

def update_state_step(new_state, step):
    '''Оновлює поточний стан та лічильник кроків'''
    state = new_state + 0
    step = step + 1
    return state, step