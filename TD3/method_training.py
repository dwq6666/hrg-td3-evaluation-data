"""Per-method training settings; shared budgets do not replace author settings."""

import numpy as np


SAC_ACTION_SAMPLING_PROTOCOL = "sac_stochastic_policy_v1"
TD3_REPAIR_LEARNING_PROTOCOL = dict(
    version="td3_repair_v2", actor_lr=1e-4, critic_lr=3e-4,
    actor_warmup_updates=500, logit_limit=2., logit_penalty=.01,
    delay_counter="cumulative_critic_updates")


def action_sampling_metadata(cfg):
    """Record behavior changes without invalidating archived, unchanged runs."""
    if cfg.algorithm == "sac":
        return {"action_sampling_protocol": SAC_ACTION_SAMPLING_PROTOCOL}
    if getattr(cfg, "network_style", None) == "original2022_gradfix":
        return {"td3_learning_protocol": dict(TD3_REPAIR_LEARNING_PROTOCOL)}
    return {}


def validate_action_sampling(cfg, metadata):
    if (cfg.algorithm == "sac" and
            metadata.get("action_sampling_protocol") != SAC_ACTION_SAMPLING_PROTOCOL):
        raise ValueError(
            "SAC checkpoint used legacy/unknown action sampling; preserve it for "
            "diagnostics and use a new run directory for stochastic-policy training"
        )
    if (getattr(cfg, "network_style", None) == "original2022_gradfix" and
            metadata.get("td3_learning_protocol") != TD3_REPAIR_LEARNING_PROTOCOL):
        raise ValueError(
            "td3_gradfix checkpoint has missing/different learning protocol; "
            "use its own fresh run directory and the versioned repair protocol"
        )


def sample_training_action(policy, cfg, args, total_steps, state, image=None):
    """SAC explores with its learned distribution; TD3/DDPG retain their noise."""
    if total_steps < args.start_timesteps:
        return np.random.uniform(-1.0, 1.0, cfg.action_dim)
    if cfg.algorithm == "sac":
        return policy.get_action(state, image, deterministic=False)
    raw = policy.get_action(state, image)
    sigma = exploration_sigma(cfg, args, total_steps)
    return np.clip(raw + np.random.normal(0.0, sigma, cfg.action_dim), -1.0, 1.0)


def training_defaults(cfg, budget):
    defaults = {name: getattr(budget, name) for name in (
        "max_timesteps", "start_timesteps", "batch_size", "eval_freq", "eval_episodes", "save_freq")}
    defaults.update(discount=0.99, expl_noise=0.15, expl_noise_min=0.02,
                    expl_decay_steps=200_000)
    if cfg.uses_original_interface:
        defaults.update(start_timesteps=0, batch_size=40, discount=0.99999,
                        expl_noise=1.0, expl_noise_min=0.1, expl_decay_steps=500_000)
    return defaults


def exploration_sigma(cfg, args, total_steps):
    step = total_steps + int(cfg.uses_original_interface)
    fraction = min(1.0, step / max(args.expl_decay_steps, 1))
    return args.expl_noise + fraction * (args.expl_noise_min - args.expl_noise)


def replay_terminal(cfg, done, timeout, reached_limit=False):
    # Author's 500-step cutoff bootstraps; existing proposed-method protocol
    # continues to treat its task deadline as a terminal failure.
    if cfg.uses_original_interface:
        return float(0 if timeout or reached_limit else done)
    return float(done or timeout)
