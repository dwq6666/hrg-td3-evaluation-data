"""Central experiment definitions for the traversability-aware TD3 project.

All training and evaluation scripts import this file.  An experiment preset is
therefore a reproducible description of the observation, network and action
ablation; users should not edit dimensions by hand in multiple scripts.
"""

from dataclasses import asdict, dataclass, replace
from typing import Dict, List


PASSABLE_HEIGHT_M = 0.025  # Height < 2.5 cm is low; equality is high.


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    algorithm: str = "td3"  # td3 | ddpg | sac | dwa
    network_style: str = "fusion"  # fusion | ral2022 | original2022 | original2022_gradfix
    reward_mode: str = "shaped"  # shaped | ral2022
    n_sectors: int = 20
    robot_dim: int = 5
    use_height: bool = False
    use_rgb: bool = False
    use_enhanced_features: bool = False
    use_sector_recal: bool = False
    recal_mode: str = "none"  # none | uniform | plain | height
    use_safety_shield: bool = False
    use_pass_gate: bool = False
    rule_pass_gate: bool = False
    pass_state_dim: int = 2
    enhanced_features_dim: int = 10
    passable_height: float = PASSABLE_HEIGHT_M
    physical_passable_height: float = PASSABLE_HEIGHT_M
    recal_hidden: int = 64
    recal_sigma: float = 0.05
    safety_stop_distance: float = 0.38
    safety_slow_distance: float = 0.75
    pass_turn_damping: float = 0.35
    pass_usage_cost: float = 0.015

    @property
    def vector_dim(self) -> int:
        dim = self.n_sectors + self.robot_dim
        if self.use_height:
            dim += self.n_sectors
        if self.use_pass_gate:
            dim += self.pass_state_dim
        if self.use_enhanced_features:
            dim += self.enhanced_features_dim
        return dim

    @property
    def action_dim(self) -> int:
        return 3 if self.use_pass_gate and not self.rule_pass_gate else 2

    @property
    def trainable(self) -> bool:
        return self.algorithm in {"td3", "ddpg", "sac"}

    @property
    def uses_original_interface(self) -> bool:
        return self.network_style in {"original2022", "original2022_gradfix"}

    def validate(self) -> None:
        valid_modes = {"none", "uniform", "plain", "height"}
        if self.algorithm not in {"td3", "ddpg", "sac", "dwa"}:
            raise ValueError("Unknown algorithm: %s" % self.algorithm)
        if self.algorithm == "ddpg" and (self.network_style != "ral2022" or
                                         self.robot_dim != 4 or self.action_dim != 2 or
                                         self.use_height or self.use_rgb):
            raise ValueError("DDPG baseline requires the matched 24-D, 2-action RA-L-style interface")
        if self.network_style not in {"fusion", "ral2022", "original2022", "original2022_gradfix"}:
            raise ValueError("Unknown network style: %s" % self.network_style)
        if self.reward_mode not in {"shaped", "ral2022"}:
            raise ValueError("Unknown reward mode: %s" % self.reward_mode)
        if self.recal_mode not in valid_modes:
            raise ValueError("Unknown recalibration mode: %s" % self.recal_mode)
        if self.use_sector_recal and not self.use_height:
            raise ValueError("Sector recalibration requires distance-height observations")
        if self.use_sector_recal and self.recal_mode == "none":
            raise ValueError("Enabled sector recalibration needs a non-'none' mode")
        if not self.use_sector_recal and self.recal_mode != "none":
            raise ValueError("Disabled sector recalibration must use mode='none'")
        if self.use_pass_gate and not self.use_safety_shield:
            raise ValueError("The pass gate is defined as an override of the safety shield")
        if self.rule_pass_gate and not self.use_pass_gate:
            raise ValueError("Rule pass gate requires pass-gate state/control")
        if (self.network_style == "ral2022" or self.uses_original_interface) and (self.robot_dim != 4 or self.action_dim != 2):
            raise ValueError("RA-L 2022 network requires 4 robot values and 2 actions")
        if self.uses_original_interface and (
                self.algorithm != "td3" or self.reward_mode != "ral2022" or
                self.use_height or self.use_rgb or self.use_safety_shield or
                self.use_pass_gate or self.use_enhanced_features):
            raise ValueError("original2022 preserves the author's distance-only TD3 and reward")
        if self.safety_slow_distance <= self.safety_stop_distance:
            raise ValueError("safety_slow_distance must exceed safety_stop_distance")

    def to_dict(self) -> Dict:
        return asdict(self)


def _cfg(name: str, description: str, **kwargs) -> ExperimentConfig:
    cfg = ExperimentConfig(name=name, description=description, **kwargs)
    cfg.validate()
    return cfg


# The presets are deliberately cumulative.  This makes attribution in the
# ablation table straightforward and prevents accidental configuration drift.
PRESETS: Dict[str, ExperimentConfig] = {
    # The paper labels original_td3 as TD3 (Baseline). Keep preset metadata
    # stable because it is stored in checkpoints and evaluation contracts.
    "original_td3": _cfg(
        "original_td3",
        "Author's public TD3 models, optimizer, observation and reward; common-scene adaptation.",
        robot_dim=4, network_style="original2022", reward_mode="ral2022",
    ),
    "td3_gradfix": _cfg(
        "td3_gradfix",
        "Experimental TD3 gradient/saturation repair with versioned stabilization; original observation, raw actions and reward.",
        robot_dim=4, network_style="original2022_gradfix", reward_mode="ral2022",
    ),
    # Historical engineering implementation; retained for old checkpoints.
    "baseline": _cfg(
        "baseline",
        "RA-L 2022 TD3 baseline: 20 distance sectors + 4 robot-state values.",
        robot_dim=4,
        network_style="ral2022",
        reward_mode="ral2022",
    ),
    "ddpg_24": _cfg(
        "ddpg_24",
        "Single-Q DDPG matched to the 24-D TD3 baseline observation, network and reward.",
        algorithm="ddpg", robot_dim=4, network_style="ral2022",
        reward_mode="ral2022",
    ),
    "height": _cfg(
        "height",
        "Distance-height joint state without recalibration or pass gate.",
        use_height=True,
    ),
    "height_uniform": _cfg(
        "height_uniform",
        "Parameter-matched sector branch with uniform weights.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="uniform",
    ),
    "height_plain": _cfg(
        "height_plain",
        "Distance-conditioned sector recalibration; height is not used by the scorer.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="plain",
    ),
    "height_recal": _cfg(
        "height_recal",
        "Proposed height-conditioned sector recalibration without pass gating.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="height",
    ),
    "height_gate": _cfg(
        "height_gate",
        "Distance-height state with learnable safety-shield override, no recalibration.",
        use_height=True,
        use_safety_shield=True,
        use_pass_gate=True,
    ),
    "height_shield": _cfg(
        "height_shield",
        "Distance-height common ablation base with a fixed safety shield; no recalibration or pass gate.",
        use_height=True,
        use_safety_shield=True,
    ),
    "full_no_gate": _cfg(
        "full_no_gate",
        "Proposed recalibration with a fixed safety shield and no override action.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="height",
        use_safety_shield=True,
    ),
    "full": _cfg(
        "full",
        "Full proposed method: height-conditioned recalibration plus learnable pass gate.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="height",
        use_safety_shield=True,
        use_pass_gate=True,
    ),
    "full_plain": _cfg(
        "full_plain",
        "Full method with distance-only recalibration scores; height observations, learned gate and shield are retained.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="plain",
        use_safety_shield=True,
        use_pass_gate=True,
    ),
    "height_rule": _cfg(
        "height_rule",
        "TD3 motion policy with a hand-coded 2.5 cm pass decision.",
        use_height=True,
        use_safety_shield=True,
        use_pass_gate=True,
        rule_pass_gate=True,
    ),
    "full_rule_gate": _cfg(
        "full_rule_gate",
        "Height-recalibrated TD3 with a fixed 2.5 cm pass rule instead of a learned gate.",
        use_height=True, use_sector_recal=True, recal_mode="height",
        use_safety_shield=True, use_pass_gate=True, rule_pass_gate=True,
    ),
    "sac_dh": _cfg(
        "sac_dh",
        "SAC with raw distance-height observation and two motion actions.",
        algorithm="sac",
        use_height=True,
    ),
    "sac_dh_gate": _cfg(
        "sac_dh_gate",
        "SAC with raw distance-height observation and learnable pass gate.",
        algorithm="sac",
        use_height=True,
        use_safety_shield=True,
        use_pass_gate=True,
    ),
    "dwa": _cfg(
        "dwa",
        "Classical Dynamic Window Approach using distance sectors.",
        algorithm="dwa",
    ),
    "height_dwa": _cfg(
        "height_dwa",
        "Height-rule Dynamic Window Approach; low sectors are traversable.",
        algorithm="dwa",
        use_height=True,
    ),
    "full_rgb": _cfg(
        "full_rgb",
        "Full method with an additional RGB branch (extension, not a core contribution).",
        use_height=True,
        use_rgb=True,
        use_sector_recal=True,
        recal_mode="height",
        use_safety_shield=True,
        use_pass_gate=True,
    ),
    "full_handcrafted": _cfg(
        "full_handcrafted",
        "Full method plus 10 handcrafted features, used only to quantify their effect.",
        use_height=True,
        use_sector_recal=True,
        recal_mode="height",
        use_safety_shield=True,
        use_pass_gate=True,
        use_enhanced_features=True,
    ),
}


# Keep the full-method interface, shield, reward and nominal architecture while
# replacing learned sector weights by ones.  This is a fresh experiment identity;
# the historical height_uniform preset has a different observation/action setup.
PRESETS["full_uniform"] = replace(
    PRESETS["full"],
    name="full_uniform",
    description=(
        "Full-method matched uniform-weight control: preserves 47-D observations, "
        "87-D augmented input, learned gate and shield. Same nominal architecture; "
        "the unused sector scorer receives no gradient."
    ),
    recal_mode="uniform",
)
PRESETS["full_uniform"].validate()


SUITES: Dict[str, List[str]] = {
    "matched_uniform": ["full", "full_uniform"],
    "pit_matched_uniform": ["full", "full_uniform"],
    "td3_repair": ["original_td3", "td3_gradfix"],
    "original_vs_ours": ["original_td3", "full"],
    "quick": ["baseline", "height_recal", "full"],
    "representation": ["height", "height_uniform", "height_plain", "height_recal"],
    "gate": ["height_recal", "full_no_gate", "height_gate", "full_rule_gate", "full"],
    "ablation": [
        "full",
        "height_gate",
        "full_no_gate",
        "full_rule_gate",
        "height_shield",
        "full_plain",
    ],
    "extensions": ["full", "full_rgb", "full_handcrafted"],
    "comparison": [
        "original_td3", "ddpg_24", "sac_dh", "sac_dh_gate",
        "dwa", "height_dwa", "height_rule", "full_rule_gate", "full",
    ],
    "paper": [
        "original_td3", "ddpg_24", "height_shield", "full_plain",
        "full_no_gate", "height_gate", "height_rule", "full_rule_gate",
        "sac_dh", "sac_dh_gate", "dwa", "height_dwa", "full",
    ],
    "threshold": ["full_h002", "full", "full_h003"],
    # The pit worlds reuse the same architecture and preset names. Signed
    # heights encode depressions, so *all* learning methods need fresh runs.
    "pit_quick": ["baseline", "full", "dwa", "height_dwa"],
    "pit_comparison": [
        "original_td3", "ddpg_24", "sac_dh", "sac_dh_gate",
        "dwa", "height_dwa", "height_rule", "full_rule_gate", "full",
    ],
    "pit_ablation": [
        "full", "height_gate", "full_no_gate", "full_rule_gate",
        "height_shield", "full_plain",
    ],
    "pit_paper": [
        "original_td3", "ddpg_24", "height_shield", "full_plain",
        "full_no_gate", "height_gate", "height_rule", "full_rule_gate",
        "sac_dh", "sac_dh_gate", "dwa", "height_dwa", "full",
    ],
}


# Threshold sensitivity changes only the policy-side h_th; the environment's
# physical classification remains 2.5 cm for every method and every run.
PRESETS["full_h002"] = _cfg(
    "full_h002", "Full method with a 2 cm recalibration threshold.",
    use_height=True, use_sector_recal=True, recal_mode="height",
    use_safety_shield=True, use_pass_gate=True, passable_height=0.02,
)
PRESETS["full_h003"] = _cfg(
    "full_h003", "Full method with a 3 cm recalibration threshold.",
    use_height=True, use_sector_recal=True, recal_mode="height",
    use_safety_shield=True, use_pass_gate=True, passable_height=0.03,
)
PRESETS["full_h005"] = _cfg(
    "full_h005", "Legacy 5 cm recalibration sensitivity, excluded from the current formal suite.",
    use_height=True, use_sector_recal=True, recal_mode="height",
    use_safety_shield=True, use_pass_gate=True, passable_height=0.05,
)


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    max_timesteps: int
    start_timesteps: int
    eval_freq: int
    eval_episodes: int
    save_freq: int
    batch_size: int = 128

    def to_dict(self) -> Dict:
        return asdict(self)


TRAINING_PROFILES: Dict[str, TrainingProfile] = {
    "smoke": TrainingProfile("smoke", 5_000, 500, 2_500, 5, 2_500, 64),
    "pilot": TrainingProfile("pilot", 200_000, 5_000, 10_000, 20, 25_000, 128),
    "final": TrainingProfile("final", 500_000, 10_000, 25_000, 50, 50_000, 256),
}


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    lidar_noise_std: float = 0.0
    height_noise_std: float = 0.0
    lidar_dropout_prob: float = 0.0
    camera_noise_std: float = 0.0
    linear_response: float = 1.0
    angular_response: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


EVALUATION_PROFILES: Dict[str, EvaluationProfile] = {
    "nominal": EvaluationProfile("nominal"),
    "sensor_noise": EvaluationProfile(
        "sensor_noise",
        lidar_noise_std=0.03,
        height_noise_std=0.01,
        lidar_dropout_prob=0.05,
        camera_noise_std=8.0,
    ),
    "dynamics_shift": EvaluationProfile(
        "dynamics_shift",
        linear_response=0.75,
        angular_response=1.20,
    ),
    "combined": EvaluationProfile(
        "combined",
        lidar_noise_std=0.03,
        height_noise_std=0.01,
        lidar_dropout_prob=0.05,
        camera_noise_std=8.0,
        linear_response=0.75,
        angular_response=1.20,
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError("Unknown preset %r. Available: %s" % (name, ", ".join(sorted(PRESETS))))


def get_evaluation_profile(name: str) -> EvaluationProfile:
    try:
        return EVALUATION_PROFILES[name]
    except KeyError:
        raise ValueError(
            "Unknown evaluation profile %r. Available: %s"
            % (name, ", ".join(sorted(EVALUATION_PROFILES)))
        )


def get_training_profile(name: str) -> TrainingProfile:
    try:
        return TRAINING_PROFILES[name]
    except KeyError:
        raise ValueError(
            "Unknown training profile %r. Available: %s"
            % (name, ", ".join(sorted(TRAINING_PROFILES)))
        )
