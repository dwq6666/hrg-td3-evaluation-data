"""Shared reward/timeout protocol for the V5 navigation experiments.

Keep this module ROS-free so the reward can be regression-tested before a
long Gazebo run. The RA-L 2022 reproduction retains its separate reward.
"""

import math

import numpy as np


PROTOCOL_VERSION = "v6_25mm_pass_2"
TIMEOUT_PENALTY = 5.0


def shaped_reward(target, collision, pit_fall, linear, angular, min_laser,
                  progress, theta, gate, pass_usage_cost):
    if target:
        return 15.0
    if collision or pit_fall:
        return -15.0
    forward = float(np.clip(linear, 0.0, 1.0))
    turn = float(np.clip(abs(angular), 0.0, 1.0))
    # Only actual progress earns the dominant positive reward. In particular,
    # facing the goal while stationary must not earn reward every time step.
    reward = 3.0 * float(np.clip(progress, -0.15, 0.15))
    reward += 0.08 * forward * max(math.cos(theta), 0.0)
    reward -= 0.03  # time cost, also penalizes zero-action deadlock
    reward -= 0.025 * turn
    reward -= 0.15 * float(np.clip((0.50 - min_laser) / 0.50, 0.0, 1.0))
    reward -= float(pass_usage_cost) * float(np.clip(gate, 0.0, 1.0))
    return float(np.clip(reward, -1.0, 1.0))


def with_timeout_penalty(reward, timeout, reward_mode):
    if timeout and reward_mode == "shaped":
        return float(reward) - TIMEOUT_PENALTY
    return float(reward)
