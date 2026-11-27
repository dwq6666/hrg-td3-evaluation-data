"""Episode-level metrics shared by training, ablation and generalization tests."""

import json
import math
import os
from typing import Dict, Optional

import numpy as np


def _divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


class MetricsCollector:
    def __init__(self):
        self.episodes = []
        self.reset_episode()

    def reset_episode(self, start_distance: float = 0.0):
        self.current = {
            "steps": 0,
            "reward": 0.0,
            "success": False,
            "collision": False,
            "pit_fall": False,
            "pit_encountered": False,
            "timeout": False,
            "task_type": "unknown",
            "start_distance": float(start_distance),
            "final_distance": float(start_distance),
            "path_length": 0.0,
            "min_clearance": float("inf"),
            "gate_sum": 0.0,
            "requested_gate_sum": 0.0,
            "requested_on_high_steps": 0,
            "requested_on_pit_steps": 0,
            "gate_veto_steps": 0,
            "gate_active_steps": 0,
            "low_encounter_steps": 0,
            "high_encounter_steps": 0,
            "gate_on_low_steps": 0,
            "gate_on_high_steps": 0,
            "pit_detected_steps": 0,
            "pit_nearby_steps": 0,
            "pit_gate_suppressed_steps": 0,
            "shield_intervention_steps": 0,
            "shield_stop_steps": 0,
            "requested_linear_sum": 0.0,
            "executed_linear_sum": 0.0,
            "absolute_angular_sum": 0.0,
            "low_motion_steps": 0,
            "shield_override_sum": 0.0,
            "low_pass_attempts": 0,
            "low_pass_successes": 0,
            "high_avoid_attempts": 0,
            "high_avoid_successes": 0,
            "attention_entropy_sum": 0.0,
            "attention_samples": 0,
        }

    def record_step(self, reward: float, done: bool, info: Optional[dict] = None):
        info = info or {}
        ep = self.current
        if ep["steps"] == 0:
            ep["task_type"] = str(info.get("task_type", "unknown"))
        ep["steps"] += 1
        ep["reward"] += float(reward)
        ep["final_distance"] = float(info.get("distance", ep["final_distance"]))
        ep["path_length"] += float(info.get("step_distance", 0.0))
        ep["min_clearance"] = min(ep["min_clearance"], float(info.get("min_laser", 10.0)))

        gate = float(info.get("pass_gate", 0.0))
        requested_gate = float(info.get("requested_gate", gate))
        gate_active = bool(info.get("gate_active", gate >= 0.5))
        front_class = info.get("front_class", "none")
        shield_scale = float(info.get("shield_scale", 1.0))
        decision_zone = shield_scale < 0.999
        ep["gate_sum"] += gate
        ep["requested_gate_sum"] += requested_gate
        ep["gate_active_steps"] += int(gate_active)
        ep["low_encounter_steps"] += int(decision_zone and front_class == "low")
        ep["high_encounter_steps"] += int(decision_zone and front_class == "high")
        ep["gate_on_low_steps"] += int(decision_zone and gate_active and front_class == "low")
        ep["gate_on_high_steps"] += int(decision_zone and gate_active and front_class == "high")
        ep["requested_on_high_steps"] += int(decision_zone and requested_gate >= 0.5 and front_class == "high")
        ep["requested_on_pit_steps"] += int(requested_gate >= 0.5 and front_class == "pit")
        ep["gate_veto_steps"] += int(info.get("gate_vetoed", False))
        ep["pit_detected_steps"] += int(info.get("pit_detected", False))
        ep["pit_nearby_steps"] += int(info.get("pit_nearby", False))
        ep["pit_gate_suppressed_steps"] += int(info.get("pit_gate_suppressed", False))
        ep["pit_encountered"] = bool(ep["pit_encountered"] or info.get("pit_nearby", False))
        ep["shield_intervention_steps"] += int(shield_scale < 0.999)
        ep["shield_stop_steps"] += int(shield_scale < 0.05)
        ep["requested_linear_sum"] += float(info.get("requested_linear", 0.0))
        ep["executed_linear_sum"] += float(info.get("executed_linear", 0.0))
        ep["absolute_angular_sum"] += abs(float(info.get("executed_angular", 0.0)))
        ep["low_motion_steps"] += int(
            float(info.get("executed_linear", 0.0)) < 0.05
            and float(info.get("step_distance", 0.0)) < 0.01
        )
        ep["shield_override_sum"] += gate * max(0.0, 1.0 - shield_scale)
        ep["low_pass_attempts"] += int(info.get("low_pass_started", False))
        ep["low_pass_successes"] += int(info.get("low_pass_success", False))
        ep["high_avoid_attempts"] += int(info.get("high_avoid_started", False))
        ep["high_avoid_successes"] += int(info.get("high_avoid_success", False))

        weights = info.get("recal_weights")
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            probabilities = weights / max(float(weights.sum()), 1e-12)
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            normalized_entropy = entropy / max(math.log(len(probabilities)), 1e-12)
            ep["attention_entropy_sum"] += normalized_entropy
            ep["attention_samples"] += 1

        if done:
            ep["success"] = bool(info.get("target", False))
            ep["collision"] = bool(info.get("collision", False))
            ep["pit_fall"] = bool(info.get("pit_fall", False))

    def finish_episode(self, timeout: bool = False):
        ep = dict(self.current)
        ep["timeout"] = bool(timeout and not ep["success"] and not ep["collision"] and not ep["pit_fall"])
        ep["min_clearance"] = 10.0 if math.isinf(ep["min_clearance"]) else ep["min_clearance"]
        if ep["success"]:
            ep["spl"] = min(1.0, ep["start_distance"] / max(ep["path_length"], ep["start_distance"], 1e-6))
        else:
            ep["spl"] = 0.0
        ep["gate_usage"] = _divide(ep["gate_active_steps"], ep["steps"])
        ep["mean_gate"] = _divide(ep["gate_sum"], ep["steps"])
        ep["attention_entropy"] = _divide(ep["attention_entropy_sum"], ep["attention_samples"])
        self.episodes.append(ep)
        return ep

    def statistics(self, last_n: Optional[int] = None) -> Dict:
        data = self.episodes[-last_n:] if last_n else self.episodes
        if not data:
            return {}

        def total(key):
            return sum(float(ep.get(key, 0.0)) for ep in data)

        def mean(key):
            return float(np.mean([float(ep.get(key, 0.0)) for ep in data]))

        successes = total("success")
        return {
            "episodes": len(data),
            "success_rate": _divide(successes, len(data)),
            "collision_rate": _divide(total("collision"), len(data)),
            "pit_fall_rate": _divide(total("pit_fall"), len(data)),
            "pit_encounter_rate": _divide(total("pit_encountered"), len(data)),
            "pit_encounter_success_rate": _divide(
                sum(float(ep.get("success", False)) for ep in data if ep.get("pit_encountered", False)),
                total("pit_encountered"),
            ),
            "pit_detected_steps": int(total("pit_detected_steps")),
            "pit_gate_suppressed_steps": int(total("pit_gate_suppressed_steps")),
            "shield_intervention_rate": _divide(total("shield_intervention_steps"), total("steps")),
            "shield_stop_rate": _divide(total("shield_stop_steps"), total("steps")),
            "mean_requested_linear": _divide(total("requested_linear_sum"), total("steps")),
            "mean_executed_linear": _divide(total("executed_linear_sum"), total("steps")),
            "mean_absolute_angular": _divide(total("absolute_angular_sum"), total("steps")),
            "low_motion_rate": _divide(total("low_motion_steps"), total("steps")),
            "timeout_rate": _divide(total("timeout"), len(data)),
            "mean_reward": mean("reward"),
            "mean_steps": mean("steps"),
            "mean_path_length": mean("path_length"),
            "mean_min_clearance": mean("min_clearance"),
            "spl": mean("spl"),
            "gate_usage_rate": _divide(total("gate_active_steps"), total("steps")),
            "mean_gate": _divide(total("gate_sum"), total("steps")),
            "mean_requested_gate": _divide(total("requested_gate_sum"), total("steps")),
            "gate_veto_rate": _divide(total("gate_veto_steps"), total("steps")),
            "gate_low_recall": _divide(total("gate_on_low_steps"), total("low_encounter_steps")),
            "gate_high_misuse_rate": _divide(total("gate_on_high_steps"), total("high_encounter_steps")),
            "gate_high_request_rate": _divide(total("requested_on_high_steps"), total("high_encounter_steps")),
            "gate_pit_request_rate": _divide(total("requested_on_pit_steps"), total("pit_detected_steps")),
            "low_pass_event_success_rate": _divide(total("low_pass_successes"), total("low_pass_attempts")),
            "high_avoid_event_success_rate": _divide(total("high_avoid_successes"), total("high_avoid_attempts")),
            "shield_override_mean": _divide(total("shield_override_sum"), total("steps")),
            "attention_entropy": _divide(total("attention_entropy_sum"), total("attention_samples")),
            "successful_episodes": int(successes),
            "collision_episodes": int(total("collision")),
            "pit_fall_episodes": int(total("pit_fall")),
            "timeout_episodes": int(total("timeout")),
        }

    def print_report(self, last_n: Optional[int] = None, title: str = "Evaluation") -> Dict:
        stats = self.statistics(last_n)
        if not stats:
            print("[%s] no completed episodes" % title)
            return stats
        print("\n%s" % ("=" * 72))
        print("%s (%d episodes)" % (title, stats["episodes"]))
        print("%s" % ("=" * 72))
        print(
            "success %.1f%% | collision %.1f%% | timeout %.1f%% | SPL %.3f"
            % (
                100.0 * stats["success_rate"],
                100.0 * stats["collision_rate"],
                100.0 * stats["timeout_rate"],
                stats["spl"],
            )
        )
        print(
            "reward %.3f | steps %.1f | path %.2fm | min clearance %.3fm"
            % (
                stats["mean_reward"],
                stats["mean_steps"],
                stats["mean_path_length"],
                stats["mean_min_clearance"],
            )
        )
        print(
            "gate use %.1f%% | low recall %.1f%% | high request/executed %.1f/%.1f%% | low-pass success %.1f%%"
            % (
                100.0 * stats["gate_usage_rate"],
                100.0 * stats["gate_low_recall"],
                100.0 * stats["gate_high_request_rate"],
                100.0 * stats["gate_high_misuse_rate"],
                100.0 * stats["low_pass_event_success_rate"],
            )
        )
        print(
            "shield %.1f%% (stop %.1f%%) | low motion %.1f%% | requested/executed v %.3f/%.3f"
            % (100.0 * stats["shield_intervention_rate"],
               100.0 * stats["shield_stop_rate"],
               100.0 * stats["low_motion_rate"],
               stats["mean_requested_linear"], stats["mean_executed_linear"])
        )
        if stats["pit_encounter_rate"] > 0 or stats["pit_fall_rate"] > 0:
            print(
                "pit encounter %.1f%% | pit fall %.1f%% | success after encounter %.1f%% | pit detections %d"
                % (100.0 * stats["pit_encounter_rate"],
                   100.0 * stats["pit_fall_rate"],
                   100.0 * stats["pit_encounter_success_rate"],
                   stats["pit_detected_steps"])
            )
        return stats

    def save(self, filepath: str, metadata: Optional[dict] = None):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "statistics": self.statistics(),
            "episodes": self.episodes,
        }
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> bool:
        """Restore completed episodes so resumed runs retain their history."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        episodes = payload.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("metrics file has an invalid episodes field: " + filepath)
        self.episodes = episodes
        self.reset_episode()
        print("[MetricsCollector] loaded %d completed episodes from %s" % (len(self.episodes), filepath))
        return True


# Removed the former hand-written BASELINE_METRICS. Every baseline comparison
# must now come from an actual run with the same seeds and evaluation profiles.
