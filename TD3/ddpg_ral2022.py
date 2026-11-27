"""Single-critic DDPG baseline matched to the 24-D RA-L-style TD3 input.

The observation, action mapping, network widths, reward and training budget
match the project's TD3 baseline. DDPG-specific differences are one Q network,
an unsmoothed target action and an actor/target update on every gradient step.
This is a common-scene implementation, not a reproduction of reported results
from the original DDPG paper.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from td3_fusion import SummaryWriter, device
from td3_ral2022 import Actor


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 800), nn.ReLU(),
            nn.Linear(800, 600), nn.ReLU(), nn.Linear(600, 1),
        )

    def forward(self, state, action):
        return self.q(torch.cat((state, action), dim=1))


class DDPG24:
    def __init__(self, vec_dim, action_dim=2, log_dir: Optional[str] = None, **_):
        if int(vec_dim) != 24 or int(action_dim) != 2:
            raise ValueError("DDPG24 requires 24 observation and 2 action values")
        self.actor = Actor(vec_dim, action_dim).to(device)
        self.actor_target = Actor(vec_dim, action_dim).to(device)
        self.critic = Critic(vec_dim, action_dim).to(device)
        self.critic_target = Critic(vec_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iter_count = self.critic_update_count = 0

    @staticmethod
    def _env_action(raw):
        return torch.cat(((raw[:, :1] + 1.0) / 2.0, raw[:, 1:]), dim=1)

    def get_action(self, state_vec, state_rgb=None):
        state = torch.as_tensor(np.asarray(state_vec).reshape(1, -1),
                                dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().ravel()

    def get_recal_weights(self):
        return None

    def train(self, replay_buffer, iterations=1, batch_size=128, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        # The last three arguments are accepted for the shared trainer API but
        # deliberately unused: DDPG has no TD3 target smoothing or delay.
        for _ in range(int(iterations)):
            state, _, action, reward, done, next_state, _ = replay_buffer.sample_batch(batch_size)
            if len(state) == 0:
                return
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            ns = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            a = torch.as_tensor(action, dtype=torch.float32, device=device)
            r = torch.as_tensor(reward, dtype=torch.float32, device=device)
            d = torch.as_tensor(done, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_action = self._env_action(self.actor_target(ns))
                target_q = r + (1.0 - d) * discount * self.critic_target(ns, next_action)
            critic_loss = F.mse_loss(self.critic(s, a), target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_update_count += 1

            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            try:
                actor_loss = -self.critic(s, self._env_action(self.actor(s))).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(True)

            with torch.no_grad():
                for source, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
                for source, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
            self.iter_count += 1
            self.writer.add_scalar("train/critic_loss", float(critic_loss.detach().cpu()), self.iter_count)
            self.writer.add_scalar("train/actor_loss", float(actor_loss.detach().cpu()), self.iter_count)
            self.writer.add_scalar("train/target_q", float(target_q.mean().cpu()), self.iter_count)

    def save_training_state(self, path, metadata=None):
        torch.save({
            "version": 1, "algorithm": "ddpg",
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "iter_count": self.iter_count,
            "critic_update_count": self.critic_update_count,
            "metadata": metadata or {},
        }, path)

    def load_training_state(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("algorithm") != "ddpg":
            raise ValueError("checkpoint is not a DDPG run")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.iter_count = int(checkpoint.get("iter_count", 0))
        self.critic_update_count = int(checkpoint.get("critic_update_count", 0))
        return checkpoint.get("metadata", {})
