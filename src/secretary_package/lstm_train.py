import torch
import torch.nn as nn
import numpy as np
from secretary_package import StepAwareSecretaryLSTM, LSTMSecretaryAgent

def train_one_episode_pg(
    env,
    agent: LSTMSecretaryAgent,
    optimizer: torch.optim.Optimizer,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip_norm: float | None = 1.0,
    gamma: float = 0.99
):
    """
    Працює з вашим кастомним API: env.step(action) -> (obs, done, info),
    а фінальний reward бере з info["reward"].
    Повертає метрики епізоду.
    """
    obs = env.reset()
    agent.reset()

    log_probs = []
    entropies = []
    values = []
    done = False
    steps = 0
    info = {}

    while not done:
        action, logp, ent, v = agent.act_train(obs)
        obs, done, info = env.step([action])

        log_probs.append(logp)
        entropies.append(ent)
        values.append(v)
        steps += 1

    # Термінальна винагорода із середовища
    reward = float(info.get("reward", 0.0))
    discounted_reward = reward * (gamma ** steps)
    R = torch.tensor(discounted_reward, dtype=torch.float32, device=agent.device)

 #   R = torch.tensor(reward, dtype=torch.float32, device=agent.device)

    log_probs_t = torch.stack(log_probs)   # (T,)
    entropies_t = torch.stack(entropies)   # (T,)
    values_t = torch.stack(values)         # (T,)

    # Advantage: R - V(s_t)
    advantages = (R - values_t).detach()

    # Policy loss: -sum logπ(a|s) * advantage
    policy_loss = -(log_probs_t * advantages).sum()

    # Value loss: MSE(V(s_t), R)
    value_loss = 0.5 * ((values_t - R) ** 2).sum()

    # Entropy bonus (мінімізуємо loss, тому віднімаємо ентропію)
    entropy_loss = -(entropies_t.sum())

    # Загальний loss
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Оновлення градієнтів
    optimizer.zero_grad()
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), grad_clip_norm)
    optimizer.step()

    return {
        "reward": reward,
        "steps": steps,
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropies_t.mean().detach().cpu().item()),
        "info": info,
    }



def train_pg(
    env,
    agent: LSTMSecretaryAgent,
    episodes: int = 10_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    print_every: int = 500,
):
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    avg_reward = 0.0
    avg_steps = 0.0
    beta = 0.98  # EMA для друку статистики

    for ep in range(1, episodes + 1):
        metrics = train_one_episode_pg(env, agent, optimizer, gamma=gamma)
        r = metrics["reward"]
        s = metrics["steps"]
        avg_reward = beta * avg_reward + (1 - beta) * r
        avg_steps = beta * avg_steps + (1 - beta) * s

        if ep % print_every == 0:
            print(
                f"ep={ep:6d} | reward={r:.4f} | ema_reward={avg_reward:.4f} | "
                f"ema_steps={avg_steps:.1f} | loss={metrics['loss']:.4f}"
               # f"steps={metrics['steps']} | loss={metrics['loss']:.4f}"
            )

    return agent


# Приклад оцінки після навчання (детермінований threshold або стохастичний sample)
def evaluate_with_simulation(run_one_side_simulation, env, agent, episodes: int = 1000):
    agent.inference_mode = "threshold"  # або "sample"
    agent.threshold = 0.5  # можна підбирати

    infos = run_one_side_simulation(env, agent, episodes=episodes)
    rewards = [i.get("reward", 0.0) for i in infos]
    mean_reward = float(np.mean(rewards)) if len(rewards) else 0.0

    print(f"Eval episodes={episodes} | mean_reward={mean_reward:.4f}")
    return infos