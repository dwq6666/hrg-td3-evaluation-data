"""Factory and action adapters shared by training and evaluation."""

import numpy as np

from sac_fusion import SACFusion
from td3_fusion import TD3Fusion
from td3_ral2022 import TD3RAL2022
from ddpg_ral2022 import DDPG24


def make_learning_policy(cfg, log_dir):
    common = dict(
        vec_dim=cfg.vector_dim, action_dim=cfg.action_dim,
        use_rgb=cfg.use_rgb, use_height=cfg.use_height,
        use_pass_gate=cfg.use_pass_gate,
        use_enhanced_features=cfg.use_enhanced_features,
        use_sector_recal=cfg.use_sector_recal, recal_mode=cfg.recal_mode,
        n_sectors=cfg.n_sectors, recal_hidden=cfg.recal_hidden,
        recal_h_th=cfg.passable_height, recal_sigma=cfg.recal_sigma,
        log_dir=log_dir,
    )
    if cfg.algorithm == "ddpg":
        return DDPG24(**common)
    if cfg.network_style == "original2022":
        from original_td3 import OriginalTD3
        return OriginalTD3(**common)
    if cfg.network_style == "original2022_gradfix":
        from td3_gradfix import GradientFixedTD3
        return GradientFixedTD3(**common)
    if cfg.network_style == "ral2022":
        return TD3RAL2022(**common)
    if cfg.algorithm == "sac":
        return SACFusion(**common)
    if cfg.algorithm == "td3":
        return TD3Fusion(**common)
    raise ValueError("%s is not a trainable policy" % cfg.algorithm)


def rule_gate_signal(state, cfg):
    """Non-learning threshold baseline using the same noisy observation."""
    n = cfg.n_sectors
    distance = np.asarray(state[:n])
    height = np.asarray(state[n:2*n])
    mid = n // 2
    front = np.arange(max(0, mid-2), min(n, mid+2))
    idx = int(front[np.argmin(distance[front])])
    should_pass = (distance[idx] < cfg.safety_slow_distance
                   and 0.0 <= height[idx] < cfg.passable_height)
    return 1.0 if should_pass else -1.0


def policy_to_env_action(raw_action, cfg, state=None):
    """Map network [-1,1] outputs to environment commands and inject rule gate."""
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    action = raw.copy()
    action[0] = (np.clip(action[0], -1.0, 1.0) + 1.0) / 2.0
    action[1:] = np.clip(action[1:], -1.0, 1.0)
    if cfg.rule_pass_gate:
        if state is None:
            raise ValueError("rule-gate action requires the current observation")
        action = np.concatenate([action[:2], [rule_gate_signal(state, cfg)]]).astype(np.float32)
    return action


def replay_action(env_action, cfg, raw_action=None):
    """Critics only receive dimensions controlled by their learning policy."""
    if cfg.uses_original_interface:
        if raw_action is None:
            raise ValueError("original_td3 replay must store the raw [-1,1] policy action")
        return np.asarray(raw_action[:cfg.action_dim], dtype=np.float32)
    return np.asarray(env_action[:cfg.action_dim], dtype=np.float32)
