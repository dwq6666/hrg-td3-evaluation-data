"""TD3 networks for distance-height traversability experiments.

The sector recalibration module never overwrites physical distance/height
channels. It appends attention-weighted, normalized channels and preserves the
raw normalized observation through a residual information path.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment_config import PASSABLE_HEIGHT_M
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard logging is optional at runtime.
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _img_batch_to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(batch_uint8), device=device)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x.float() / 255.0


def _img_to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(img_uint8), device=device)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.float() / 255.0


class VectorPreprocessor(nn.Module):
    """Deterministic scale normalization that keeps the state layout intact."""

    def __init__(
        self,
        n_sectors: int,
        use_height: bool,
        use_pass_gate: bool,
        use_enhanced_features: bool,
        distance_scale: float = 10.0,
        height_scale: float = 0.50,
        goal_distance_scale: float = 20.0,
        progress_scale: float = 0.10,
    ):
        super().__init__()
        self.n = int(n_sectors)
        self.use_height = bool(use_height)
        self.use_pass_gate = bool(use_pass_gate)
        self.use_enhanced_features = bool(use_enhanced_features)
        self.distance_scale = float(distance_scale)
        self.height_scale = float(height_scale)
        self.goal_distance_scale = float(goal_distance_scale)
        self.progress_scale = float(progress_scale)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        x = vec.clone()
        n = self.n
        x[:, :n] = torch.clamp(x[:, :n] / self.distance_scale, 0.0, 1.0)
        offset = n
        if self.use_height:
            x[:, offset : offset + n] = torch.clamp(
                x[:, offset : offset + n] / self.height_scale, -2.0, 2.0
            )
            offset += n

        # Robot state: distance, relative angle, executed v, executed w, progress.
        x[:, offset] = torch.clamp(x[:, offset] / self.goal_distance_scale, 0.0, 1.5)
        x[:, offset + 1] = torch.clamp(x[:, offset + 1] / math.pi, -1.0, 1.0)
        x[:, offset + 2] = torch.clamp(x[:, offset + 2], 0.0, 1.5)
        x[:, offset + 3] = torch.clamp(x[:, offset + 3], -1.5, 1.5)
        x[:, offset + 4] = torch.clamp(x[:, offset + 4] / self.progress_scale, -2.0, 2.0)
        offset += 5

        if self.use_pass_gate:
            x[:, offset : offset + 2] = torch.clamp(x[:, offset : offset + 2], -1.0, 1.0)
            offset += 2

        if self.use_enhanced_features:
            x[:, offset : offset + 3] = torch.clamp(
                x[:, offset : offset + 3] / self.height_scale, -2.0, 2.0
            )
            x[:, offset + 6 : offset + 9] = torch.clamp(
                x[:, offset + 6 : offset + 9] / self.distance_scale, 0.0, 1.0
            )
        return x


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, out_dim)

    def forward(self, img_bchw: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(img_bchw))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc(x.flatten(1)))


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(vec))))


class SectorRecalibration(nn.Module):
    """Direction-selective recalibration for distance-height sector pairs.

    ``uniform``, ``plain`` and ``height`` instantiate identical parameters.
    The plain ablation masks height-derived scorer inputs, while the height mode
    uses height difference and criticality around the passability threshold.
    """

    def __init__(
        self,
        n_sectors: int = 20,
        hidden: int = 64,
        h_th_normalized: float = 0.08,
        sigma_normalized: float = 0.10,
        mode: str = "height",
    ):
        super().__init__()
        if mode not in {"uniform", "plain", "height"}:
            raise ValueError("Unsupported recalibration mode: %s" % mode)
        self.n = int(n_sectors)
        self.mode = str(mode)
        self.h_th = float(h_th_normalized)
        self.sigma = max(float(sigma_normalized), 1e-6)
        self.local_encoder = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(2 * hidden, 1)
        self.last_weights: Optional[torch.Tensor] = None

    @property
    def appended_dim(self) -> int:
        return 2 * self.n

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        n = self.n
        distance = vec[:, :n]
        height = vec[:, n : 2 * n]
        if self.mode == "uniform":
            weights = torch.ones_like(distance)
        else:
            if self.mode == "plain":
                scorer_height = torch.zeros_like(height)
                dh = torch.zeros_like(height)
                abs_dh = torch.zeros_like(height)
                criticality = torch.zeros_like(height)
            else:
                scorer_height = height
                dh = height - self.h_th
                abs_dh = dh.abs()
                criticality = torch.exp(-((dh / self.sigma) ** 2))
            tokens = torch.stack(
                [distance, scorer_height, dh, abs_dh, criticality], dim=-1
            )
            local = self.local_encoder(tokens)
            context = self.context_encoder(local.mean(dim=1))
            context = context.unsqueeze(1).expand(-1, n, -1)
            logits = self.score(torch.cat([local, context], dim=-1)).squeeze(-1)
            weights = torch.softmax(logits, dim=1) * float(n)

        self.last_weights = weights.detach()
        return torch.cat([vec, distance * weights, height * weights], dim=1)


class StateEncoder(nn.Module):
    def __init__(
        self,
        vec_dim: int,
        use_rgb: bool,
        use_height: bool,
        use_pass_gate: bool,
        use_enhanced_features: bool,
        use_sector_recal: bool,
        recal_mode: str,
        n_sectors: int,
        recal_hidden: int,
        recal_h_th: float,
        recal_sigma: float,
    ):
        super().__init__()
        self.use_rgb = bool(use_rgb)
        self.preprocess = VectorPreprocessor(
            n_sectors=n_sectors,
            use_height=use_height,
            use_pass_gate=use_pass_gate,
            use_enhanced_features=use_enhanced_features,
        )
        self.sector_recal = None
        encoded_vec_dim = int(vec_dim)
        if use_sector_recal:
            if not use_height:
                raise ValueError("Sector recalibration requires height observations")
            self.sector_recal = SectorRecalibration(
                n_sectors=n_sectors,
                hidden=recal_hidden,
                h_th_normalized=recal_h_th / self.preprocess.height_scale,
                sigma_normalized=recal_sigma / self.preprocess.height_scale,
                mode=recal_mode,
            )
            encoded_vec_dim += self.sector_recal.appended_dim
        self.vec_enc = VectorEncoder(encoded_vec_dim, 256)
        self.img_enc = ImageEncoder(256) if self.use_rgb else None
        self.out_dim = 512 if self.use_rgb else 256

    def forward(self, vec: torch.Tensor, image: Optional[torch.Tensor]) -> torch.Tensor:
        vec = self.preprocess(vec)
        if self.sector_recal is not None:
            vec = self.sector_recal(vec)
        vector_features = self.vec_enc(vec)
        if not self.use_rgb:
            return vector_features
        if image is None:
            raise ValueError("RGB-enabled network received no image batch")
        return torch.cat([vector_features, self.img_enc(image)], dim=1)


class ActorFusion(nn.Module):
    def __init__(self, vec_dim: int, action_dim: int, **encoder_kwargs):
        super().__init__()
        self.encoder = StateEncoder(vec_dim=vec_dim, **encoder_kwargs)
        self.fc1 = nn.Linear(self.encoder.out_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    @property
    def sector_recal(self):
        return self.encoder.sector_recal

    def forward(self, vec: torch.Tensor, image: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.encoder(vec, image)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticFusion(nn.Module):
    def __init__(self, vec_dim: int, action_dim: int, **encoder_kwargs):
        super().__init__()
        self.encoder = StateEncoder(vec_dim=vec_dim, **encoder_kwargs)
        in_dim = self.encoder.out_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, vec: torch.Tensor, image: Optional[torch.Tensor], action: torch.Tensor):
        features = self.encoder(vec, image)
        x = torch.cat([features, action], dim=1)
        return self.q1(x), self.q2(x)


class TD3Fusion:
    def __init__(
        self,
        vec_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        use_rgb: bool = False,
        use_height: bool = True,
        use_pass_gate: bool = True,
        use_enhanced_features: bool = False,
        use_sector_recal: bool = True,
        recal_mode: str = "height",
        n_sectors: int = 20,
        recal_hidden: int = 64,
        recal_h_th: float = PASSABLE_HEIGHT_M,
        recal_sigma: float = 0.05,
        log_dir: Optional[str] = None,
    ):
        encoder_kwargs = dict(
            use_rgb=use_rgb,
            use_height=use_height,
            use_pass_gate=use_pass_gate,
            use_enhanced_features=use_enhanced_features,
            use_sector_recal=use_sector_recal,
            recal_mode=recal_mode,
            n_sectors=n_sectors,
            recal_hidden=recal_hidden,
            recal_h_th=recal_h_th,
            recal_sigma=recal_sigma,
        )
        self.use_rgb = bool(use_rgb)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.actor = ActorFusion(vec_dim, action_dim, **encoder_kwargs).to(device)
        self.actor_target = ActorFusion(vec_dim, action_dim, **encoder_kwargs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = CriticFusion(vec_dim, action_dim, **encoder_kwargs).to(device)
        self.critic_target = CriticFusion(vec_dim, action_dim, **encoder_kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iter_count = 0
        self.critic_update_count = 0

    def _prepare_single_image(self, image):
        return _img_to_tensor(image) if self.use_rgb else None

    def _prepare_image_batch(self, images):
        return _img_batch_to_tensor(images) if self.use_rgb else None

    @staticmethod
    def _actor_to_env_action(raw_action: torch.Tensor) -> torch.Tensor:
        linear = (raw_action[:, 0:1] + 1.0) / 2.0
        return torch.cat([linear, raw_action[:, 1:]], dim=1)

    def get_action(self, state_vec: np.ndarray, state_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        vec = torch.as_tensor(state_vec.reshape(1, -1), dtype=torch.float32, device=device)
        image = self._prepare_single_image(state_rgb)
        with torch.no_grad():
            return self.actor(vec, image).cpu().numpy().flatten()

    def get_recal_weights(self) -> Optional[np.ndarray]:
        module = self.actor.sector_recal
        if module is None or module.last_weights is None:
            return None
        return module.last_weights[0].cpu().numpy().copy()

    def train(
        self,
        replay_buffer,
        iterations: int = 1,
        batch_size: int = 128,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ) -> None:
        mean_loss = 0.0
        mean_q = 0.0
        for _ in range(int(iterations)):
            state_vec, state_rgb, actions, rewards, dones, next_vec, next_rgb = replay_buffer.sample_batch(batch_size)
            if len(state_vec) == 0:
                return
            state_vec_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)
            next_vec_t = torch.as_tensor(next_vec, dtype=torch.float32, device=device)
            state_img_t = self._prepare_image_batch(state_rgb)
            next_img_t = self._prepare_image_batch(next_rgb)
            action_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
            reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_raw = self.actor_target(next_vec_t, next_img_t)
                noise = (torch.randn_like(next_raw) * policy_noise).clamp(-noise_clip, noise_clip)
                next_raw = (next_raw + noise).clamp(-self.max_action, self.max_action)
                next_action = self._actor_to_env_action(next_raw)
                target_q1, target_q2 = self.critic_target(next_vec_t, next_img_t, next_action)
                target_q = reward_t + (1.0 - done_t) * discount * torch.min(target_q1, target_q2)

            current_q1, current_q2 = self.critic(state_vec_t, state_img_t, action_t)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()
            self.critic_update_count += 1

            if self.critic_update_count % int(policy_freq) == 0:
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(False)
                actor_raw = self.actor(state_vec_t, state_img_t)
                actor_action = self._actor_to_env_action(actor_raw)
                actor_q1, _ = self.critic(state_vec_t, state_img_t, actor_action)
                actor_loss = -actor_q1.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(True)
                with torch.no_grad():
                    for source, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target.mul_(1.0 - tau).add_(source, alpha=tau)
                    for source, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target.mul_(1.0 - tau).add_(source, alpha=tau)

            mean_loss += float(critic_loss.detach().cpu())
            mean_q += float(target_q.mean().detach().cpu())

        self.iter_count += int(iterations)
        self.writer.add_scalar("train/critic_loss", mean_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/target_q", mean_q / iterations, self.iter_count)

    def save_training_state(self, path: str, metadata: Optional[dict] = None) -> None:
        torch.save(
            {
                "version": 2,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "iter_count": self.iter_count,
                "critic_update_count": self.critic_update_count,
                "metadata": metadata or {},
            },
            path,
        )

    def load_training_state(self, path: str, load_optimizers: bool = True) -> dict:
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.iter_count = int(checkpoint.get("iter_count", 0))
        self.critic_update_count = int(checkpoint.get("critic_update_count", 0))
        return checkpoint.get("metadata", {})

    def save(self, filename: str, directory: str) -> None:
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename: str, directory: str) -> None:
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename), map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
