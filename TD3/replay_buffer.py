"""Replay buffer with optional RGB storage.

Distance-only ablations do not store dummy 84x84 images, which keeps a large
ablation suite practical. RGB frames are stored as uint8 when enabled.
"""

import os
import pickle
import random
from collections import deque

import numpy as np


class ReplayBufferRGB:
    def __init__(self, buffer_size, random_seed=123, use_rgb=True):
        self.buffer_size = int(buffer_size)
        self.use_rgb = bool(use_rgb)
        self.buffer = deque(maxlen=self.buffer_size)
        self._rng = random.Random(int(random_seed))

    @property
    def count(self):
        return len(self.buffer)

    def add(self, state_vec, state_rgb, action, reward, terminal, next_vec, next_rgb):
        experience = (
            np.asarray(state_vec, dtype=np.float32),
            np.asarray(state_rgb, dtype=np.uint8) if self.use_rgb else None,
            np.asarray(action, dtype=np.float32),
            float(reward),
            float(terminal),
            np.asarray(next_vec, dtype=np.float32),
            np.asarray(next_rgb, dtype=np.uint8) if self.use_rgb else None,
        )
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if not self.buffer:
            empty = np.empty((0,), dtype=np.float32)
            return empty, None, empty, empty, empty, empty, None
        batch = self._rng.sample(self.buffer, min(int(batch_size), len(self.buffer)))
        states = np.stack([item[0] for item in batch], axis=0)
        images = np.stack([item[1] for item in batch], axis=0) if self.use_rgb else None
        actions = np.stack([item[2] for item in batch], axis=0)
        rewards = np.asarray([item[3] for item in batch], dtype=np.float32).reshape(-1, 1)
        terminals = np.asarray([item[4] for item in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.stack([item[5] for item in batch], axis=0)
        next_images = np.stack([item[6] for item in batch], axis=0) if self.use_rgb else None
        return states, images, actions, rewards, terminals, next_states, next_images

    def clear(self):
        self.buffer.clear()

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": 2,
            "buffer": list(self.buffer),
            "buffer_size": self.buffer_size,
            "use_rgb": self.use_rgb,
            "random_state": self._rng.getstate(),
        }
        with open(filepath, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("[ReplayBuffer] saved %d transitions to %s" % (self.size(), filepath))

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        with open(filepath, "rb") as handle:
            payload = pickle.load(handle)
        if bool(payload.get("use_rgb", True)) != self.use_rgb:
            raise ValueError("Replay-buffer RGB setting does not match the experiment preset")
        loaded = payload["buffer"]
        self.buffer = deque(loaded[-self.buffer_size :], maxlen=self.buffer_size)
        if "random_state" in payload:
            self._rng.setstate(payload["random_state"])
        print("[ReplayBuffer] loaded %d transitions from %s" % (self.size(), filepath))
        return True


# Compatibility alias for older helper scripts.
ReplayBuffer = ReplayBufferRGB
