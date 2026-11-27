"""Evaluate a saved preset under nominal or shifted test conditions."""

import argparse
import hashlib
import csv
import json
import os
import random

import numpy as np
import torch

from experiment_config import EVALUATION_PROFILES, PRESETS, get_evaluation_profile, get_preset
from classical_baselines import DynamicWindowController, load_controller_parameters
from metrics_collector import MetricsCollector
from policy_factory import make_learning_policy, policy_to_env_action
from scenario_manifest import load_or_create
from velodyne_env import GazeboRGBEnv
from arena_config import (arena_id, world_sha256, expected_training_world,
                          is_evaluation_only, verify_versioned_world, validate_checkpoint_origin)
from navigation_reward import PROTOCOL_VERSION, with_timeout_penalty
from method_training import validate_action_sampling


HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--seed", type=int, default=0, help="training seed/run identifier")
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--profile", choices=sorted(EVALUATION_PROFILES), default="nominal")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--run-root", default=os.path.join(HERE, "experiments"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--launch-sim", action="store_true")
    parser.add_argument("--launchfile", default="paper_traversability.launch")
    parser.add_argument("--robot-name", default="p3dx")
    parser.add_argument("--trace-episodes", type=int, default=0,
                        help="save per-step control and perception diagnostics for the first N episodes")
    parser.add_argument('--controller-params', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    profile = get_evaluation_profile(args.profile)
    run_name = args.run_name or "%s_seed%d" % (cfg.name, args.seed)
    run_dir = os.path.abspath(os.path.join(args.run_root, run_name))
    scene_id = os.path.splitext(os.path.basename(args.launchfile))[0]
    arena = arena_id(args.launchfile)
    verify_versioned_world(args.launchfile)
    if cfg.trainable and args.controller_params:
        raise ValueError('Controller parameters apply only to non-learning baselines')
    with_pits = arena in ("paper_v4", "paper_v5", "paper_v6", "paper_pits_v3")
    checkpoint = os.path.abspath(args.checkpoint or os.path.join(run_dir, "checkpoint.pt"))
    if cfg.trainable and not os.path.exists(checkpoint):
        raise FileNotFoundError("checkpoint not found: " + checkpoint)
    # Scene I and II deliberately use the same new fixed episode poses; the
    # changed obstacle layout, not changed starts/goals, is the test variable.
    manifest_name = "scenarios_%s_seed%d.json" % (scene_id if is_evaluation_only(args.launchfile) else arena, args.eval_seed)
    scenario_file = args.scenario_file or os.path.join(args.run_root, "benchmarks", manifest_name)
    scenarios = load_or_create(scenario_file, args.episodes, args.eval_seed,
                               with_pits=with_pits, arena=arena, launchfile=args.launchfile)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    env = GazeboRGBEnv(
        args.launchfile, cfg.n_sectors, robot_name=args.robot_name,
        connect_only=not args.launch_sim,
        use_height=cfg.use_height, use_rgb=cfg.use_rgb,
        use_pass_gate=cfg.use_pass_gate, use_safety_shield=cfg.use_safety_shield,
        use_enhanced_features=cfg.use_enhanced_features,
        robot_dim=cfg.robot_dim, reward_mode=cfg.reward_mode,
        passable_height=cfg.physical_passable_height,
        safety_stop_distance=cfg.safety_stop_distance,
        safety_slow_distance=cfg.safety_slow_distance,
        pass_turn_damping=cfg.pass_turn_damping,
        pass_usage_cost=cfg.pass_usage_cost,
        lidar_noise_std=profile.lidar_noise_std,
        height_noise_std=profile.height_noise_std,
        lidar_dropout_prob=profile.lidar_dropout_prob,
        camera_noise_std=profile.camera_noise_std,
        linear_response=profile.linear_response,
        angular_response=profile.angular_response,
        seed=args.eval_seed,
        use_original_observation=cfg.uses_original_interface,
    )
    if cfg.trainable:
        policy = make_learning_policy(cfg, os.path.join(run_dir, "tensorboard_eval"))
        metadata = policy.load_training_state(checkpoint, load_optimizers=False)
        validate_action_sampling(cfg, metadata)
        if metadata.get("preset") not in (None, cfg.name):
            raise ValueError("checkpoint was trained with preset %r, not %r" % (metadata.get("preset"), cfg.name))
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("checkpoint uses an older training protocol; retrain in a new run-root")
        validate_checkpoint_origin(metadata,args.launchfile)
    else:
        parameters, parameter_file_hash = load_controller_parameters(args.controller_params, cfg.name)
        policy = DynamicWindowController(cfg, parameters=parameters)
        metadata = {"preset": cfg.name, "controller": "DWA reimplementation",
                    'controller_parameters': policy.parameters,
                    'parameter_file_sha256': parameter_file_hash}

    print("Evaluating %s / %s for %d episodes" % (cfg.name, profile.name, args.episodes))
    metrics = MetricsCollector()
    trace_fields = (
        "episode", "step", "robot_x", "robot_y", "robot_yaw", "robot_roll", "robot_pitch",
        "goal_x", "goal_y", "distance", "step_distance", "front_class",
        "front_distance", "front_height", "pit_detected",
        "requested_linear", "requested_angular", "shield_scale", "pass_signal",
        "pass_gate", "executed_linear", "executed_angular", "target", "collision", "pit_fall",
    )
    trace_file = None
    trace_writer = None
    trace_path = os.path.join(
        run_dir, "trace_%s_%s_seed%d.csv" % (scene_id, profile.name, args.eval_seed)
    )
    if args.trace_episodes > 0:
        trace_file = open(trace_path, "w", newline="", encoding="utf-8")
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fields)
        trace_writer.writeheader()
    for episode in range(args.episodes):
        state, image = env.reset(scenario=scenarios[episode])
        metrics.reset_episode(env.prev_distance)
        ended = False
        for step in range(args.max_episode_steps):
            raw = policy.get_action(state, image)
            weights = policy.get_recal_weights() if cfg.trainable else None
            action = policy_to_env_action(raw, cfg, state) if cfg.trainable else raw
            state, image, reward, done, info = env.step(action)
            timeout = not done and step + 1 >= args.max_episode_steps
            reward = with_timeout_penalty(reward, timeout, cfg.reward_mode)
            if weights is not None:
                info["recal_weights"] = weights
            metrics.record_step(reward, done, info)
            if trace_writer is not None and episode < args.trace_episodes:
                trace_writer.writerow({"episode": episode, "step": step, **{
                    key: info.get(key, "") for key in trace_fields if key not in ("episode", "step")
                }})
            if done:
                ended = True
                break
        metrics.finish_episode(timeout=not ended)
        if (episode + 1) % 10 == 0:
            metrics.print_report(last_n=10, title="Episodes %d-%d" % (episode - 8, episode + 1))

    if trace_file is not None:
        trace_file.close()
        print("Trace saved: %s" % trace_path)

    stats = metrics.print_report(title="%s / %s" % (cfg.name, profile.name))
    output = os.path.join(run_dir, "evaluation_%s_%s_seed%d.json" % (scene_id, profile.name, args.eval_seed))
    metrics.save(output, {
        "preset": cfg.name, "training_seed": args.seed if cfg.trainable else None,
        "evaluation_seed": args.eval_seed, "profile": profile.to_dict(),
        "scene_id": scene_id, "scenario_file": os.path.abspath(scenario_file),
        "arena_id": arena, "world_sha256": world_sha256(args.launchfile),
        'evaluation_only_layout': is_evaluation_only(args.launchfile),
        'source_training_world_sha256': expected_training_world(args.launchfile),
        "protocol_version": metadata.get("protocol_version", "legacy") if cfg.trainable
                            else PROTOCOL_VERSION,
        "checkpoint": checkpoint if cfg.trainable else None, "checkpoint_metadata": metadata,
        "evaluation_contract_version": 1,
        "method_config": cfg.to_dict(),
        "evaluation_episodes": len(scenarios),
        "max_episode_steps": args.max_episode_steps,
        "scenario_sha256": hashlib.sha256(json.dumps(
            scenarios, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    })
    if cfg.trainable:
        policy.writer.close()
    print("Saved: %s" % output)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
