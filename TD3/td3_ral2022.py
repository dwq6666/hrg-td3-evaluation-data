"""RA-L 2022 TD3 network/reward-compatible baseline adapted to common I/O."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from td3_fusion import SummaryWriter, device


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 800)
        self.l2 = nn.Linear(800, 600)
        self.l3 = nn.Linear(600, action_dim)

    def forward(self, state):
        return torch.tanh(self.l3(F.relu(self.l2(F.relu(self.l1(state))))))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 800), nn.ReLU(), nn.Linear(800, 600), nn.ReLU(), nn.Linear(600, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 800), nn.ReLU(), nn.Linear(800, 600), nn.ReLU(), nn.Linear(600, 1))

    def forward(self, state, action):
        value = torch.cat([state, action], dim=1)
        return self.q1(value), self.q2(value)


class TD3RAL2022:
    def __init__(self, vec_dim, action_dim=2, max_action=1.0, log_dir: Optional[str] = None, **_):
        self.action_dim, self.max_action = int(action_dim), float(max_action)
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
        return torch.cat([(raw[:, :1] + 1.0) / 2.0, raw[:, 1:]], dim=1)

    def get_action(self, state_vec, state_rgb=None):
        state = torch.as_tensor(np.asarray(state_vec).reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().ravel()

    def get_recal_weights(self):
        return None

    def train(self, replay_buffer, iterations=1, batch_size=128, discount=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        losses, values = 0.0, 0.0
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
                raw = self.actor_target(ns)
                noise = (torch.randn_like(raw) * policy_noise).clamp(-noise_clip, noise_clip)
                na = self._env_action((raw + noise).clamp(-1, 1))
                tq1, tq2 = self.critic_target(ns, na)
                target = r + (1-d) * discount * torch.min(tq1, tq2)
            q1, q2 = self.critic(s, a)
            loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optimizer.zero_grad(); loss.backward(); self.critic_optimizer.step()
            self.critic_update_count += 1
            if self.critic_update_count % policy_freq == 0:
                actor_loss = -self.critic(s, self._env_action(self.actor(s)))[0].mean()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
                with torch.no_grad():
                    for source, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.mul_(1-tau).add_(source, alpha=tau)
                    for source, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.mul_(1-tau).add_(source, alpha=tau)
            losses += float(loss.detach().cpu()); values += float(target.mean().cpu())
        self.iter_count += int(iterations)
        self.writer.add_scalar("train/critic_loss", losses / iterations, self.iter_count)
        self.writer.add_scalar("train/target_q", values / iterations, self.iter_count)

    def save_training_state(self, path, metadata=None):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(), "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "iter_count": self.iter_count, "critic_update_count": self.critic_update_count,
            "metadata": metadata or {},
        }, path)

    def load_training_state(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"]); self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.iter_count = int(checkpoint.get("iter_count", 0))
        self.critic_update_count = int(checkpoint.get("critic_update_count", 0))
        return checkpoint.get("metadata", {})
