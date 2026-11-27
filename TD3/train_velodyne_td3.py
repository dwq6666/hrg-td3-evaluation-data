"""Train one reproducible traversability-ablation preset.

Example:
  python3 train_velodyne_td3.py --preset full --seed 0
"""

import argparse
import json
import os
import pickle
import random
import signal
import time
import numpy as np
import torch

from experiment_config import PRESETS, TRAINING_PROFILES, get_preset, get_training_profile
from metrics_collector import MetricsCollector
from policy_factory import make_learning_policy, policy_to_env_action, replay_action
from replay_buffer import ReplayBufferRGB
from velodyne_env import GazeboRGBEnv
from arena_config import arena_id, is_v5_scene2, is_evaluation_only, world_sha256
from navigation_reward import PROTOCOL_VERSION, with_timeout_penalty
from method_training import (training_defaults, exploration_sigma, replay_terminal,
                             sample_training_action, action_sampling_metadata,
                             validate_action_sampling)
from training_snapshot import (commit_snapshot, resolve_snapshot, publish_exports,
                               has_checkpoint, training_lock, atomic_json)


HERE = os.path.dirname(os.path.abspath(__file__))


def raw_to_env_action(raw, cfg=None, state=None):
    """Compatibility wrapper; new code should pass cfg for rule-gate presets."""
    if cfg is not None:
        return policy_to_env_action(raw, cfg, state)
    action = np.asarray(raw, dtype=np.float32).copy()
    action[0] = (np.clip(action[0], -1.0, 1.0) + 1.0) / 2.0
    action[1:] = np.clip(action[1:], -1.0, 1.0)
    return action


def make_policy(cfg, log_dir):
    return make_learning_policy(cfg, log_dir)


def update_original_episode(policy, replay, args, steps):
    """Finish an episode's gradient batch before honoring Ctrl-C."""
    interrupted = [False]
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def defer_interrupt(signum, frame):
        interrupted[0] = True

    for sig in previous_handlers:
        signal.signal(sig, defer_interrupt)
    try:
        policy.train(replay, iterations=steps, batch_size=args.batch_size,
                     discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip, policy_freq=args.policy_freq)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return interrupted[0]


def make_env(cfg, args, seed):
    env = GazeboRGBEnv(
        args.launchfile,
        cfg.n_sectors,
        robot_name=args.robot_name,
        connect_only=not args.launch_sim,
        use_height=cfg.use_height,
        use_rgb=cfg.use_rgb,
        use_pass_gate=cfg.use_pass_gate,
        use_safety_shield=cfg.use_safety_shield,
        use_enhanced_features=cfg.use_enhanced_features,
        robot_dim=cfg.robot_dim,
        reward_mode=cfg.reward_mode,
        passable_height=cfg.physical_passable_height,
        safety_stop_distance=cfg.safety_stop_distance,
        safety_slow_distance=cfg.safety_slow_distance,
        pass_turn_damping=cfg.pass_turn_damping,
        pass_usage_cost=cfg.pass_usage_cost,
        domain_randomization=args.domain_randomization,
        seed=seed,
        use_original_observation=cfg.uses_original_interface,
    )
    if env.observation_dim != cfg.vector_dim:
        raise RuntimeError("environment/network dimension mismatch: %d != %d" % (env.observation_dim, cfg.vector_dim))
    return env


def evaluate(policy, env, cfg, episodes, max_steps, seed):
    """Deterministic in-training evaluation; restore the training RNG afterward."""
    saved_rng = env.get_rng_state()
    saved_training_step = env.training_step
    env.training_step = None
    env.seed(seed)
    metrics = MetricsCollector()
    for _ in range(episodes):
        state, image = env.reset()
        metrics.reset_episode(env.prev_distance)
        ended = False
        for step in range(max_steps):
            raw = policy.get_action(state, image)
            weights = policy.get_recal_weights()
            action = raw_to_env_action(raw, cfg, state)
            state, image, reward, done, info = env.step(action)
            timeout = not done and step + 1 >= max_steps
            reward = with_timeout_penalty(reward, timeout, cfg.reward_mode)
            if weights is not None:
                info["recal_weights"] = weights
            metrics.record_step(reward, done, info)
            if done:
                ended = True
                break
        metrics.finish_episode(timeout=not ended)
    env.set_rng_state(saved_rng)
    env.training_step = saved_training_step
    return metrics


def save_run(policy, replay, metrics, run_dir, cfg, args, total_steps, pending_updates=0, env=None):
    metadata = {
        "preset": cfg.name,
        "seed": args.seed,
        "total_steps": int(total_steps),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.to_dict(),
        "arguments": vars(args),
        "arena_id": arena_id(args.launchfile),
        "world_sha256": world_sha256(args.launchfile),
        "protocol_version": PROTOCOL_VERSION,
        "pending_episode_updates": int(pending_updates),
        "replay_size": int(replay.size()),
    }
    metadata.update(action_sampling_metadata(cfg))
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
           "env": env.get_rng_state() if env is not None and hasattr(env, 'get_rng_state') else None}
    def save_rng(path, _metadata):
        with open(path, 'wb') as handle:
            pickle.dump(rng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    writers = {"checkpoint.pt": policy.save_training_state,
               "train_metrics.json": metrics.save, "rng.pkl": save_rng}
    if args.save_buffer:
        writers["replay.pkl"] = lambda path, _metadata: replay.save(path)
    # A second Ctrl-C/SIGTERM during serialization must not interrupt a graceful
    # save. A power interruption still leaves the previous generation recoverable.
    interrupted = [False]
    def defer(signum, frame):
        interrupted[0] = True
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, defer)
        result = commit_snapshot(run_dir, metadata, writers)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if interrupted[0]:
        raise KeyboardInterrupt
    return result


def restore_rng(path, env):
    if not os.path.exists(path):
        print('[Snapshot] legacy run has no full RNG state; simulator restarts a new episode.')
        return
    with open(path, 'rb') as handle:
        rng = pickle.load(handle)
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    if rng.get('env') is not None:
        env.set_rng_state(rng['env'])


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", choices=sorted(TRAINING_PROFILES), default="final")
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--start-timesteps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.20)
    parser.add_argument("--noise-clip", type=float, default=0.50)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--expl-noise", type=float, default=None)
    parser.add_argument("--expl-noise-min", type=float, default=None)
    parser.add_argument("--expl-decay-steps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--run-root", default=os.path.join(HERE, "experiments"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing checkpoint")
    parser.add_argument("--save-buffer", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--launch-sim", action="store_true", help="start roscore/Gazebo in this process")
    parser.add_argument("--launchfile", default="paper_traversability.launch")
    parser.add_argument("--robot-name", default="p3dx")
    return parser.parse_args()


def main():
    args = parse_args()
    if is_v5_scene2(args.launchfile) or is_evaluation_only(args.launchfile):
        raise ValueError("V5 Scene II is held out for evaluation; train on paper_traversability.launch")
    cfg = get_preset(args.preset)
    if not cfg.trainable:
        raise ValueError("preset %s is a non-learning controller and needs evaluation only" % cfg.name)
    budget = get_training_profile(args.budget)
    for name, value in training_defaults(cfg, budget).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    original = cfg.uses_original_interface
    run_name = args.run_name or "%s_seed%d" % (cfg.name, args.seed)
    run_dir = os.path.abspath(os.path.join(args.run_root, run_name))
    os.makedirs(run_dir, exist_ok=True)
    with training_lock(run_dir):
        train(args, cfg, run_dir)


def train(args, cfg, run_dir):
    original = cfg.uses_original_interface
    checkpoint = os.path.join(run_dir, "checkpoint.pt")
    if has_checkpoint(run_dir) and not args.resume and not args.overwrite:
        raise FileExistsError(
            "run already exists; use --resume, --overwrite, or a different --run-name: " + run_dir
        )
    if args.overwrite and any(os.path.exists(os.path.join(run_dir, name)) for name in
                             ('snapshot_index.json', 'snapshot_index.previous.json')):
        raise ValueError('Preserve managed snapshots; use a new --run-name instead of --overwrite')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\nPreset: %s\n%s" % (cfg.name, cfg.description))
    print("vector_dim=%d action_dim=%d RGB=%s run=%s" % (cfg.vector_dim, cfg.action_dim, cfg.use_rgb, run_dir))
    env = make_env(cfg, args, args.seed)
    policy = make_policy(cfg, os.path.join(run_dir, "tensorboard"))
    replay = ReplayBufferRGB(args.buffer_size, args.seed, use_rgb=cfg.use_rgb)
    train_metrics = MetricsCollector()
    total_steps = 0
    if args.resume:
        snapshot = resolve_snapshot(run_dir, require_replay=args.save_buffer)
        publish_exports(run_dir, snapshot)
        checkpoint = str(snapshot['directory'] / 'checkpoint.pt')
        metadata = policy.load_training_state(checkpoint, load_optimizers=True)
        if metadata != snapshot['metadata']:
            raise ValueError('Model and metrics do not belong to the same saved generation')
        validate_action_sampling(cfg, metadata)
        if metadata.get("preset") not in (None, cfg.name):
            raise ValueError("checkpoint preset does not match --preset")
        if (metadata.get("arena_id") != arena_id(args.launchfile) or
                metadata.get("world_sha256") != world_sha256(args.launchfile)):
            raise ValueError("cannot resume: checkpoint arena or world file differs from this run")
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("cannot resume: reward/timeout protocol changed; start a new run-root")
        if metadata.get('seed') != args.seed or metadata.get('config') != cfg.to_dict():
            raise ValueError('cannot resume: training seed or method configuration differs')
        total_steps = int(metadata.get("total_steps", 0))
        buffer_path = str(snapshot['directory'] / "replay.pkl")
        if args.save_buffer and total_steps > 0 and not os.path.exists(buffer_path):
            raise FileNotFoundError(
                "cannot resume a --save-buffer run without its replay.pkl: " + buffer_path
            )
        if os.path.exists(buffer_path):
            replay.load(buffer_path)
            if metadata.get('replay_size') is not None and replay.size() != metadata['replay_size']:
                raise ValueError('Replay size differs from committed snapshot metadata')
        restore_rng(str(snapshot['directory'] / 'rng.pkl'), env)
        pending = int(metadata.get("pending_episode_updates", 0))
        if original and pending:
            if not os.path.exists(buffer_path):
                raise FileNotFoundError("original_td3 pending episode updates require replay.pkl")
            policy.train(replay, iterations=pending, batch_size=args.batch_size,
                         discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                         noise_clip=args.noise_clip, policy_freq=args.policy_freq)
        metrics_path = str(snapshot['directory'] / "train_metrics.json")
        if os.path.exists(metrics_path):
            train_metrics.load(metrics_path)
        if snapshot['legacy']:
            # Seal the checked legacy triplet before performing more training.
            save_run(policy, replay, train_metrics, run_dir, cfg, args, total_steps, env=env)

    atomic_json(os.path.join(run_dir, "config.json"),
                  {"method": cfg.to_dict(), "training": vars(args),
                   **action_sampling_metadata(cfg),
                   "protocol_version": PROTOCOL_VERSION})

    env.training_step = total_steps
    state, image = env.reset()
    train_metrics.reset_episode(env.prev_distance)
    episode_steps = 0
    count_rand_actions, random_action = 0, None
    last_eval = last_save = total_steps
    try:
        while total_steps < args.max_timesteps:
            raw = sample_training_action(policy, cfg, args, total_steps, state, image)
            if original:
                # Retain the author's near-obstacle exploration and sector slice.
                if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1:
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)
                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    raw = random_action.copy()
                    raw[0] = -1
            weights = policy.get_recal_weights()
            action = raw_to_env_action(raw, cfg, state)
            next_state, next_image, reward, done, info = env.step(action)
            if weights is not None:
                info["recal_weights"] = weights
            episode_steps += 1
            timeout = (not done) and episode_steps >= args.max_episode_steps
            reward = with_timeout_penalty(reward, timeout, cfg.reward_mode)
            replay.add(state, image, replay_action(action, cfg, raw_action=raw), reward,
                       replay_terminal(cfg, done, timeout,
                                       reached_limit=episode_steps >= args.max_episode_steps),
                       next_state, next_image)
            train_metrics.record_step(reward, done, info)
            state, image = next_state, next_image
            total_steps += 1

            if not original and replay.size() >= args.batch_size and total_steps >= args.start_timesteps:
                policy.train(
                    replay, iterations=1, batch_size=args.batch_size,
                    discount=args.discount, tau=args.tau,
                    policy_noise=args.policy_noise, noise_clip=args.noise_clip,
                    policy_freq=args.policy_freq,
                )

            episode_ended = done or timeout
            if episode_ended:
                train_metrics.finish_episode(timeout=timeout)
                if original:
                    interrupted = update_original_episode(policy, replay, args, episode_steps)
                    episode_steps = 0
                    if interrupted:
                        raise KeyboardInterrupt
                if len(train_metrics.episodes) % 10 == 0:
                    train_metrics.print_report(last_n=10, title="Train @ step %d" % total_steps)
                env.training_step = total_steps
                state, image = env.reset()
                train_metrics.reset_episode(env.prev_distance)
                episode_steps = 0

            if (args.eval_freq > 0 and total_steps - last_eval >= args.eval_freq
                    and (not original or episode_ended)):
                evaluation = evaluate(policy, env, cfg, args.eval_episodes, args.max_episode_steps, args.seed + 100_000)
                evaluation.print_report(title="Nominal eval @ step %d" % total_steps)
                evaluation.save(
                    os.path.join(run_dir, "eval_step_%09d.json" % total_steps),
                    {"preset": cfg.name, "seed": args.seed, "total_steps": total_steps,
                     **action_sampling_metadata(cfg),
                     "protocol_version": PROTOCOL_VERSION},
                )
                env.training_step = total_steps
                state, image = env.reset()
                train_metrics.reset_episode(env.prev_distance)
                episode_steps = 0
                last_eval = total_steps

            if (args.save_freq > 0 and total_steps - last_save >= args.save_freq
                    and (not original or episode_ended)):
                save_run(policy, replay, train_metrics, run_dir, cfg, args, total_steps, env=env)
                last_save = total_steps
    except KeyboardInterrupt:
        print("\nInterrupted; saving a resumable checkpoint...")
    finally:
        save_run(policy, replay, train_metrics, run_dir, cfg, args, total_steps,
                 pending_updates=episode_steps if original else 0, env=env)
        policy.writer.close()
    print("Training finished at step %d. Results: %s" % (total_steps, run_dir))


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    main()
