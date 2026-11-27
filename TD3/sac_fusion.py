"""Soft Actor-Critic baseline using the same observation encoder and protocol."""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_config import PASSABLE_HEIGHT_M
from td3_fusion import StateEncoder, SummaryWriter, _img_batch_to_tensor, _img_to_tensor, device


class GaussianActor(nn.Module):
    def __init__(self, vec_dim, action_dim, **encoder_kwargs):
        super().__init__()
        self.encoder = StateEncoder(vec_dim=vec_dim, **encoder_kwargs)
        self.body = nn.Sequential(nn.Linear(self.encoder.out_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    @property
    def sector_recal(self):
        return self.encoder.sector_recal

    def distribution(self, state, image):
        features = self.body(self.encoder(state, image))
        return self.mean(features), self.log_std(features).clamp(-20.0, 2.0)

    def sample(self, state, image, deterministic=False):
        mean, log_std = self.distribution(state, image)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        raw = torch.tanh(pre_tanh)
        env_action = torch.cat([(raw[:, :1] + 1.0) / 2.0, raw[:, 1:]], dim=1)
        if deterministic:
            return raw, env_action, None
        normal_log_prob = -0.5 * (((pre_tanh - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = normal_log_prob.sum(dim=1, keepdim=True)
        log_prob -= torch.log(1.0 - raw.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        log_prob -= math.log(0.5)  # first action is affinely mapped to [0, 1]
        return raw, env_action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, vec_dim, action_dim, **encoder_kwargs):
        super().__init__()
        self.encoder = StateEncoder(vec_dim=vec_dim, **encoder_kwargs)
        dim = self.encoder.out_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, image, action):
        encoded = self.encoder(state, image)
        value = torch.cat([encoded, action], dim=1)
        return self.q1(value), self.q2(value)


class SACFusion:
    def __init__(self, vec_dim, action_dim, use_rgb=False, use_height=True,
                 use_pass_gate=False, use_enhanced_features=False,
                 use_sector_recal=False, recal_mode="none", n_sectors=20,
                 recal_hidden=64, recal_h_th=PASSABLE_HEIGHT_M, recal_sigma=0.05,
                 log_dir: Optional[str] = None, **_):
        kwargs = dict(
            use_rgb=use_rgb, use_height=use_height, use_pass_gate=use_pass_gate,
            use_enhanced_features=use_enhanced_features,
            use_sector_recal=use_sector_recal, recal_mode=recal_mode,
            n_sectors=n_sectors, recal_hidden=recal_hidden,
            recal_h_th=recal_h_th, recal_sigma=recal_sigma,
        )
        self.use_rgb, self.action_dim = bool(use_rgb), int(action_dim)
        self.actor = GaussianActor(vec_dim, action_dim, **kwargs).to(device)
        self.critic = TwinCritic(vec_dim, action_dim, **kwargs).to(device)
        self.critic_target = TwinCritic(vec_dim, action_dim, **kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(action_dim)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iter_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _image(self, image):
        return _img_to_tensor(image) if self.use_rgb else None

    def _images(self, images):
        return _img_batch_to_tensor(images) if self.use_rgb else None

    def get_action(self, state_vec, state_rgb=None, deterministic=True):
        """Use deterministic actions for evaluation, distribution samples for training."""
        state = torch.as_tensor(np.asarray(state_vec).reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            raw, _, _ = self.actor.sample(state, self._image(state_rgb), deterministic=deterministic)
        return raw.cpu().numpy().ravel()

    def get_recal_weights(self):
        module = self.actor.sector_recal
        if module is None or module.last_weights is None:
            return None
        return module.last_weights[0].cpu().numpy().copy()

    def train(self, replay_buffer, iterations=1, batch_size=256, discount=0.99,
              tau=0.005, **_):
        mean_critic = mean_actor = mean_alpha = 0.0
        for _ in range(int(iterations)):
            state, images, action, reward, done, next_state, next_images = replay_buffer.sample_batch(batch_size)
            if len(state) == 0:
                return
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            ns = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            a = torch.as_tensor(action, dtype=torch.float32, device=device)
            r = torch.as_tensor(reward, dtype=torch.float32, device=device)
            d = torch.as_tensor(done, dtype=torch.float32, device=device)
            si, nsi = self._images(images), self._images(next_images)
            with torch.no_grad():
                _, next_action, next_logp = self.actor.sample(ns, nsi)
                tq1, tq2 = self.critic_target(ns, nsi, next_action)
                target = r + (1-d) * discount * (torch.min(tq1, tq2) - self.alpha.detach() * next_logp)
            q1, q2 = self.critic(s, si, a)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

            _, sampled_action, logp = self.actor.sample(s, si)
            aq1, aq2 = self.critic(s, si, sampled_action)
            actor_loss = (self.alpha.detach() * logp - torch.min(aq1, aq2)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
            with torch.no_grad():
                for source, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.mul_(1-tau).add_(source, alpha=tau)
            mean_critic += float(critic_loss.detach().cpu())
            mean_actor += float(actor_loss.detach().cpu())
            mean_alpha += float(self.alpha.detach().cpu())
        self.iter_count += int(iterations)
        self.writer.add_scalar("train/critic_loss", mean_critic / iterations, self.iter_count)
        self.writer.add_scalar("train/actor_loss", mean_actor / iterations, self.iter_count)
        self.writer.add_scalar("train/alpha", mean_alpha / iterations, self.iter_count)

    def save_training_state(self, path, metadata=None):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(), "iter_count": self.iter_count,
            "metadata": metadata or {},
        }, path)

    def load_training_state(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        self.log_alpha.data.copy_(checkpoint.get("log_alpha", torch.tensor(0.0, device=device)))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.iter_count = int(checkpoint.get("iter_count", 0))
        return checkpoint.get("metadata", {})
