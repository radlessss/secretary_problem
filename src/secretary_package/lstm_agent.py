import torch
import torch.nn as nn

class StepAwareSecretaryLSTM(nn.Module):
    """
    Online policy для задачі секретаря. 
    Вхід на кроці (one-side): x_t = [t_norm, max_so_far, current_quality] 
    shape: (batch, 3)
    Вихід:
    - stop_logit: (batch, 1) логіт для Bernoulli (STOP=1) 
    - value: (batch, 1) оцінка V(s_t) як baseline 
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 64):
        super().__init__() # Виклик super().__init__() обов'язковий для реєстрації параметрів 
        self.hidden_size = hidden_size
        
        # LSTMCell для online обробки кроків 
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Голова політики: STOP vs CONTINUE (Bernoulli) 
        self.policy_head = nn.Linear(hidden_size, 1)
        
        # Голова цінності: baseline V(s) 
        self.value_head = nn.Linear(hidden_size, 1)

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None):
        """
        Ініціалізує (h, c) нулями. [cite: 74]
        LSTMCell очікує h і c розміром (batch, hidden_size). 
        """
        if device is None:
            # Визначаємо пристрій на основі параметрів моделі 
            device = next(self.parameters()).device
            
        h = torch.zeros(batch_size, self.hidden_size, device=device) 
        c = torch.zeros(batch_size, self.hidden_size, device=device) 
        return (h, c)

    def forward_step(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """
        Один online крок. 
        x_t: (batch, input_size) 
        state: (h, c) де кожен (batch, hidden_size) 
        """
        # Обробка поточного входу через LSTMCell [cite: 86]
        h, c = self.lstm_cell(x_t, state)
        
        stop_logit = self.policy_head(h) # (batch, 1) 
        value = self.value_head(h)       # (batch, 1) 
        
        return stop_logit, value, (h, c) 




class LSTMSecretaryAgent:
    def __init__(
        self,
        model: StepAwareSecretaryLSTM,
        device: str | torch.device = "cpu",
        inference_mode: str = "sample", # "sample" або "threshold"
        threshold: float = 0.5,
    ):
        self.model = model
        self.device = torch.device(device)
        self.inference_mode = inference_mode
        self.threshold = float(threshold)
        self.state = None
        self.model.to(self.device)

    def reset(self):
        """Ініціалізує внутрішній стан LSTM на початку епізоду."""
        self.state = self.model.reset_state(batch_size=1, device=self.device)

    def _to_tensor_obs(self, obs) -> torch.Tensor:
        """
        Нормалізує формат obs для моделі:
        - якщо env повертає [obs] (список із одним елементом), беремо obs[0].
        - якщо повертає np.array shape (3,), робимо (1,3).
        """
        if isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
            
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        return obs_t

    @torch.no_grad()
    def make_decision(self, obs) -> int:
        """
        Інференс-метод для run_one_side_simulation.
        """
        self.model.eval()
        x_t = self._to_tensor_obs(obs)
        
        # Отримуємо логіт зупинки, оцінку стану та оновлюємо пам'ять LSTM 
        stop_logit, value, self.state = self.model.forward_step(x_t, self.state)
        
        if self.inference_mode == "threshold":
            prob_stop = torch.sigmoid(stop_logit) # Перетворюємо логіт у ймовірність 
            action = int((prob_stop.item() >= self.threshold))
            return action
            
        # Режим "sample" для стохастичного вибору дії
        dist = torch.distributions.Bernoulli(logits=stop_logit)
        action = dist.sample()
        return int(action.item())

    def act_train(self, obs):
        """
        Тренувальний крок: семпл дії + log_prob + entropy + value.
        Використовується для розрахунку loss у policy gradient.
        """
        self.model.train()
        x_t = self._to_tensor_obs(obs)
        
        stop_logit, value, self.state = self.model.forward_step(x_t, self.state)
        
        # Bernoulli(logits=...) стабільно працює з log_prob через logits-форму
        dist = torch.distributions.Bernoulli(logits=stop_logit)
        action = dist.sample()             # Семплимо дію (0 або 1)
        log_prob = dist.log_prob(action)   # Необхідно для обчислення градієнта політики 
        entropy = dist.entropy()           # Використовується для ентропійного бонусу 
        
        return int(action.item()), log_prob.squeeze(), entropy.squeeze(), value.squeeze()