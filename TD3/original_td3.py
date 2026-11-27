"""Resume/logging adapter for the author's TD3; learning code stays upstream."""

import numpy as np
import torch

from original_td3_models import Actor, Critic, TD3
from td3_fusion import SummaryWriter, device


IMPLEMENTATION_ID = "cimurs_public_td3_50b550d5_common_scene_v1"


class OriginalTD3(TD3):
    def __init__(self, vec_dim, action_dim=2, log_dir=None, **_):
        # Same models, target initialization and Adam defaults as the author.
        self.actor = Actor(vec_dim, action_dim).to(device)
        self.actor_target = Actor(vec_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(vec_dim, action_dim).to(device)
        self.critic_target = Critic(vec_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = 1.0
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iter_count = 0

    def get_action(self, state_vec, state_rgb=None):
        return super().get_action(np.asarray(state_vec))

    def get_recal_weights(self):
        return None

    def train(self, replay_buffer, iterations=1, batch_size=40, discount=0.99999,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        class BufferAdapter:
            def sample_batch(self, count):
                state, _, action, reward, done, next_state, _ = replay_buffer.sample_batch(count)
                return state, action, reward, done, next_state

        if iterations > 0 and replay_buffer.size() > 0:
            return super().train(BufferAdapter(), iterations, batch_size, discount,
                                 tau, policy_noise, noise_clip, policy_freq)

    def save_training_state(self, path, metadata=None):
        torch.save({
            "implementation_id": IMPLEMENTATION_ID,
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "iter_count": self.iter_count, "metadata": metadata or {},
        }, path)

    def load_training_state(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError("original_td3 requires its own fresh checkpoints; baseline is incompatible")
        for name in ("actor", "critic", "actor_target", "critic_target"):
            getattr(self, name).load_state_dict(checkpoint[name])
        if load_optimizers:
            for name in ("actor_optimizer", "critic_optimizer"):
                getattr(self, name).load_state_dict(checkpoint[name])
        self.iter_count = int(checkpoint.get("iter_count", 0))
        return checkpoint.get("metadata", {})
