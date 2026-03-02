import torch
import torch.nn as nn
import numpy as np
from secretary_package import StepAwareSecretaryLSTM, LSTMSecretaryAgent

def train_one_episode_two_sided(
    env,
    agent_a: LSTMSecretaryAgent,
    agent_b: LSTMSecretaryAgent,
    optimizer_a: torch.optim.Optimizer,
    optimizer_b: torch.optim.Optimizer,
    value_coef: float = 0.5,
    entropy_coef: float = 0.05, # Трішки вище для MARL, щоб уникнути колапсу стратегій
    grad_clip_norm: float = 1.0,
):
    obs = env.reset() # Очікується [obs_a, obs_b]
    agent_a.reset()
    agent_b.reset()

    # Списки для обох агентів
    trajectories = {
        'a': {'log_probs': [], 'entropies': [], 'values': []},
        'b': {'log_probs': [], 'entropies': [], 'values': []}
    }
    
    done = False
    steps = 0
    info = {}

    while not done:
        # Обидва агенти роблять вибір на основі своїх спостережень
        action_a, logp_a, ent_a, v_a = agent_a.act_train(obs[0])
        action_b, logp_b, ent_b, v_b = agent_b.act_train(obs[1])

        # Крок у середовищі: передаємо список дій
        obs, done, info = env.step([action_a, action_b])

        # Зберігаємо дані траєкторії
        trajectories['a']['log_probs'].append(logp_a)
        trajectories['a']['entropies'].append(ent_a)
        trajectories['a']['values'].append(v_a)
        
        trajectories['b']['log_probs'].append(logp_b)
        trajectories['b']['entropies'].append(ent_b)
        trajectories['b']['values'].append(v_b)
        
        steps += 1

    # Винагорода (зазвичай спільна за успішний матч)
    reward_val = float(info.get("reward", 0.0))
    R = torch.tensor(reward_val, dtype=torch.float32, device=agent_a.device)

    # Функція обчислення loss для одного агента
    def get_loss(traj, reward_tensor):
        log_probs_t = torch.stack(traj['log_probs'])
        values_t = torch.stack(traj['values'])
        entropies_t = torch.stack(traj['entropies'])
        
        advantages = (reward_tensor - values_t).detach()
        policy_loss = -(log_probs_t * advantages).sum()
        value_loss = 0.5 * ((values_t - reward_tensor) ** 2).sum()
        entropy_loss = -(entropies_t.sum())
        
        return policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Обчислюємо втрати
    loss_a = get_loss(trajectories['a'], R)
    loss_b = get_loss(trajectories['b'], R)

    # Оновлюємо агента А
    optimizer_a.zero_grad()
    loss_a.backward()
    if grad_clip_norm: torch.nn.utils.clip_grad_norm_(agent_a.model.parameters(), grad_clip_norm)
    optimizer_a.step()

    # Оновлюємо агента Б
    optimizer_b.zero_grad()
    loss_b.backward()
    if grad_clip_norm: torch.nn.utils.clip_grad_norm_(agent_b.model.parameters(), grad_clip_norm)
    optimizer_b.step()

    return {"reward": reward_val, "steps": steps, "loss_a": loss_a.item(), "loss_b": loss_b.item()}


def train_two_sided_pg(
    env,
    episodes: int = 10_000,
    lr: float = 3e-4,
    print_every: int = 500
):
    # 1. Створюємо моделі
    model_a = StepAwareSecretaryLSTM(input_size=3, hidden_size=64)
    model_b = StepAwareSecretaryLSTM(input_size=3, hidden_size=64)
    
    # 2. Створюємо агентів
    agent_a = LSTMSecretaryAgent(model_a, device="cpu")
    agent_b = LSTMSecretaryAgent(model_b, device="cpu")
    
    # 3. Оптимізатори (окремі для кожної моделі)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)
    
    avg_reward = 0.0
    beta = 0.99

    print(f"Starting Two-Sided training for {episodes} episodes...")

    for ep in range(1, episodes + 1):
        metrics = train_one_episode_two_sided(env, agent_a, agent_b, opt_a, opt_b)
        
        r = metrics["reward"]
        avg_reward = beta * avg_reward + (1 - beta) * r

        if ep % print_every == 0:
            print(f"Ep {ep:5d} | EMA Reward: {avg_reward:.4f} | Steps: {metrics['steps']} | "
                  f"L_A: {metrics['loss_a']:.2f} | L_B: {metrics['loss_b']:.2f}")

    return agent_a, agent_b

# --- ЗАПУСК ---
# env = YourTwoSidedSecretaryEnv(...)
# agent_a, agent_b = train_two_sided_pg(env, episodes=5000)