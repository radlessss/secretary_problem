import torch
import torch.nn as nn

class StepAwareSecretaryLSTM(nn.Module):
    """
    LSTM-based policy для задачі секретаря (one-side).

    Вхід:
        x_t = [t, max_so_far, current_quality]
        shape: (batch, 3)

    Вихід:
        stop_logit : (batch, 1) — логіт для Bernoulli (STOP = 1)
        value      : (batch, 1) — оцінка стану V(s_t)
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        self.policy_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None):
        """
        Ініціалізація стану LSTM (h, c) нулями.
        Returns:
            (h, c): tuple тензорів форми (batch, hidden_size)
        """

        if device is None:
            device = next(self.parameters()).device
            
        h = torch.zeros(batch_size, self.hidden_size, device=device) 
        c = torch.zeros(batch_size, self.hidden_size, device=device) 

        return h, c

    def forward_step(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """
        Один крок обробки.
        Args:
            x_t   : (batch, input_size)
            state : (h, c)
        Returns:
            stop_logit, value, (h, c)
        """
        h, c = self.lstm_cell(x_t, state)
        
        stop_logit = self.policy_head(h) 
        value = self.value_head(h)      
        
        return stop_logit, value, (h, c) 
    
    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        return self.forward_step(x_t, state)




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
        """Ініціалізація внутрішнього стану LSTM на початку епізоду."""
        self.state = self.model.reset_state(batch_size=1, device=self.device)

    def _to_tensor_obs(self, obs) -> torch.Tensor:
        """Перетворення observation у tensor форми (1, input_size)"""
        if isinstance(obs, (list, tuple)) and len(obs) == 1:
            obs = obs[0]
            
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

        return obs_t

    @torch.no_grad()
    def make_decision(self, obs) -> int:
        """Інференс (без градієнтів)."""

        self.model.eval()
        x_t = self._to_tensor_obs(obs)
        
        stop_logit, value, self.state = self.model.forward_step(x_t, self.state)
        
        # deterministic mode
        if self.inference_mode == "threshold":
            prob_stop = torch.sigmoid(stop_logit) 
            action = int((prob_stop.item() >= self.threshold))
            return action
            
        # stochastic mode
        dist = torch.distributions.Bernoulli(logits=stop_logit)
        action = dist.sample()

        return int(action.item())

    def act_train(self, obs):
        """
        Тренувальний крок.
        Returns:
            action, log_prob, entropy, value
        """

        self.model.train()
        x_t = self._to_tensor_obs(obs)
        
        stop_logit, value, self.state = self.model.forward_step(x_t, self.state)
        
        dist = torch.distributions.Bernoulli(logits=stop_logit)
        action = dist.sample()            
        log_prob = dist.log_prob(action)  
        entropy = dist.entropy()          
        
        return int(action.item()), log_prob.squeeze(), entropy.squeeze(), value.squeeze()