import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass
class DQNConfig:
    learning_rate: float = 0.001
    gamma: float = 0.9
    epsilon: float = 0.9
    batch_size: int = 16
    experience_pool_size: int = 512
    epochs: int = 1
    target_update_freq: int = 10

    def __post_init__(self):
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real)
                or not isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be a finite positive number")
        
        self.learning_rate = float(self.learning_rate)

        for name in ("gamma", "epsilon"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
            setattr(self, name, float(value))

        for name in ("batch_size", "experience_pool_size", "epochs", "target_update_freq"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))

        if self.batch_size > self.experience_pool_size:
            raise ValueError("batch_size must not exceed experience_pool_size")


class ExperiencePool:
    def __init__(self, state_dim, capacity=512):
        if isinstance(state_dim, bool) or not isinstance(state_dim, Integral) or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)

    def push(self, state, action, reward, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs]
        )

    def __len__(self):
        return self.size


class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config=None, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = DQNConfig() if config is None else config
        self.device = torch.device(device)

        self.q_val = _QNetwork(state_dim, action_dim).to(self.device)
        self.q_tar = _QNetwork(state_dim, action_dim).to(self.device)
        self.q_tar.load_state_dict(self.q_val.state_dict())
        self.q_tar.eval()

        self.optimizer = optim.Adam(self.q_val.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ExperiencePool(state_dim=self.state_dim, capacity=self.config.experience_pool_size)
        self.learn_step_counter = 0

    @staticmethod
    def compute_reward(old_f, new_f):
        old_arr = np.asarray(old_f, dtype=float)
        new_arr = np.asarray(new_f, dtype=float)

        new_dominates_old = np.all(new_arr <= old_arr) and np.any(new_arr < old_arr)
        if new_dominates_old:
            return 5

        old_dominates_new = np.all(old_arr <= new_arr) and np.any(old_arr < new_arr)
        if not old_dominates_new and not np.allclose(old_arr, new_arr):
            return 10

        return 0

    def select_action(self, state):
        if random.random() > self.config.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_val.eval()
        with torch.no_grad():
            q_values = self.q_val(state_tensor)
        self.q_val.train()

        return int(torch.argmax(q_values, dim=1).item())

    def store_and_learn(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)
        if len(self.memory) < self.config.experience_pool_size:
            return

        for _ in range(self.config.epochs):
            states, actions, rewards, next_states = self.memory.sample(self.config.batch_size)
            s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            s_next = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

            q_pred = self.q_val(s_t).gather(1, a_t).squeeze(1)
            with torch.no_grad():
                q_next = self.q_tar(s_next).max(dim=1)[0]
                q_target = r_t + self.config.gamma * q_next

            loss = self.loss_fn(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.learn_step_counter += 1
            if self.learn_step_counter % self.config.target_update_freq == 0:
                self.q_tar.load_state_dict(self.q_val.state_dict())
