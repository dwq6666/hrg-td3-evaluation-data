"""Gazebo environment for traversability-aware navigation experiments.

The pass action is a continuous override of a distance-only safety shield. It
has real control authority but receives no oracle height label: the policy must
learn when overriding is useful (low obstacle) or dangerous (high obstacle).
"""

import math
import os
import subprocess
import time
from typing import Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from pit_perception import PitMemory, signed_sector_observation, vertical_row_from_quaternion
from training_curriculum import open_task_probability, sample_v5_open_task
from pass_gate_logic import authorized_pass_gate, unsafe_pass_request_cost
from navigation_reward import shaped_reward
from original_observation import update_sector as update_original_sector
from experiment_config import PASSABLE_HEIGHT_M
from arena_config import (V4_TASKS, V5_TASKS, V6_TASKS, arena_id, near_v4_pit,
                          near_v5_pit, near_v6_pit, valid_v4_position,
                          valid_v5_position, valid_v6_position,
                          verify_versioned_world, is_evaluation_only,
                          near_evaluation_pit, valid_evaluation_position,
                          layout_pit_tasks)

GOAL_REACHED_DIST = 0.30
COLLISION_DIST = 0.35
TIME_DELTA = 0.10
PASSABLE_OBSTACLE_HEIGHT = PASSABLE_HEIGHT_M
SENSOR_HEIGHT = 0.25
SENSOR_X_OFFSET = 0.125
GROUND_TOL = 0.01
FREE_DIST = 5.0
PIT_DEPTH_TOL = 0.11
PIT_WIDTH = 0.90
PIT_NEAR_EDGE_OFFSET = PIT_WIDTH
PIT_FALL_Z = -0.08
PIT_CENTRES = {
    1: ((-2.2, 1.75), (2.2, -1.75)),
    2: ((-2.2, -1.75), (2.2, 1.75)),
}
_PIT_SPAWN_MASKS = tuple(
    (x - 0.85, x + 0.85, y - 0.85, y + 0.85)
    for x in (-2.2, 2.2) for y in (-1.75, 1.75)
)


def _inside_rect(x, y, x_min, x_max, y_min, y_max):
    return x_min <= x <= x_max and y_min <= y <= y_max


_HAZARD_REGIONS = (
    # Conservative common masks for paper_traversability Scene I and II.
    # Goals, robot spawns and movable boxes must not start on fixed obstacles.
    (-6.15, -3.05, 2.60, 5.50), (-2.45, 0.45, 2.60, 5.50),
    (0.55, 3.65, 2.55, 5.55), (3.35, 6.35, 2.55, 5.55),
    (-6.45, -3.70, -1.55, 1.55), (3.25, 6.25, -1.55, 1.55),
    (-6.10, -3.00, -5.00, -2.20), (-1.95, 1.15, -5.00, -2.15),
    (2.90, 6.00, -5.00, -2.10),
)
_SPECIAL_TERRAIN_REGIONS = ()


def _inside_any_region(x, y, regions):
    return any(_inside_rect(x, y, *region) for region in regions)


def is_valid_goal_position(x, y, with_pits=False):
    return (abs(x) <= 6.60 and abs(y) <= 5.60
            and not _inside_any_region(x, y, _HAZARD_REGIONS)
            and not (with_pits and _inside_any_region(x, y, _PIT_SPAWN_MASKS)))


def is_safe_spawn_position(x, y, with_pits=False):
    return is_valid_goal_position(x, y, with_pits) and not _inside_any_region(x, y, _SPECIAL_TERRAIN_REGIONS)


def is_valid_dynamic_obstacle_position(x, y, with_pits=False):
    return is_safe_spawn_position(x, y, with_pits) and not _inside_rect(x, y, -1.0, 1.0, -1.0, 1.0)


def check_pos(x, y):
    return is_valid_goal_position(x, y)


def _resize_rgb(rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    try:
        import cv2  # type: ignore
        return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    except Exception:
        try:
            from PIL import Image as PILImage  # type: ignore
            return np.asarray(PILImage.fromarray(rgb).resize((out_w, out_h)), dtype=np.uint8)
        except Exception:
            return rgb


class GazeboRGBEnv:
    """Velodyne navigation environment with optional height/RGB/pass channels."""

    def __init__(
        self, launchfile: str, environment_dim: int,
        camera_topic: str = "/r1/front_camera/image_raw",
        image_size_hw: Tuple[int, int] = (84, 84), *, robot_name: str = "p3dx",
        odom_topic: Optional[str] = None, cmd_vel_topic: Optional[str] = None,
        connect_only: bool = True, ros_port: str = "11311",
        require_topics: bool = True, startup_wait_s: float = 3.0,
        use_height: bool = True, use_rgb: bool = False,
        use_pass_gate: bool = True, use_safety_shield: bool = True,
        use_enhanced_features: bool = False,
        robot_dim: int = 5, reward_mode: str = "shaped",
        passable_height: float = PASSABLE_OBSTACLE_HEIGHT,
        safety_stop_distance: float = 0.38, safety_slow_distance: float = 0.75,
        pass_turn_damping: float = 0.35, pass_usage_cost: float = 0.015,
        lidar_noise_std: float = 0.0, height_noise_std: float = 0.0,
        lidar_dropout_prob: float = 0.0, camera_noise_std: float = 0.0,
        linear_response: float = 1.0, angular_response: float = 1.0,
        domain_randomization: bool = False, seed: int = 0,
        use_original_observation: bool = False,
    ):
        self.environment_dim = int(environment_dim)
        self.robot_name = robot_name
        self.odom_topic = odom_topic or "/%s/odom" % robot_name
        self.cmd_vel_topic = cmd_vel_topic or "/%s/cmd_vel" % robot_name
        if camera_topic == "/r1/front_camera/image_raw" and robot_name != "r1":
            camera_topic = "/%s/front_camera/image_raw" % robot_name
        self.image_size_hw = image_size_hw

        self.use_height = bool(use_height)
        self.use_rgb = bool(use_rgb)
        self.use_pass_gate = bool(use_pass_gate)
        self.use_safety_shield = bool(use_safety_shield)
        self.use_enhanced_features = bool(use_enhanced_features)
        self.robot_dim = int(robot_dim)
        self.reward_mode = str(reward_mode)
        self.use_original_observation = bool(use_original_observation)
        self.original_velodyne_data = np.full(self.environment_dim, 10.0, dtype=np.float32)
        if self.robot_dim not in (4, 5):
            raise ValueError("robot_dim must be 4 or 5")
        if self.reward_mode not in ("shaped", "ral2022"):
            raise ValueError("reward_mode must be shaped or ral2022")
        self.passable_height = float(passable_height)
        launch_name = os.path.basename(launchfile)
        self.arena_id = arena_id(launch_name)
        verify_versioned_world(launchfile)
        self.evaluation_launchfile = launch_name if is_evaluation_only(launch_name) else None
        self.with_pits = self.arena_id in ("paper_v4", "paper_v5", "paper_v6", "paper_pits_v3")
        self.fixed_pit_arena = self.arena_id in ("paper_v4", "paper_v5", "paper_v6")
        self.pit_tasks = (layout_pit_tasks(self.evaluation_launchfile) if self.evaluation_launchfile else
                          V6_TASKS if self.arena_id == "paper_v6" else
                          V5_TASKS if self.arena_id == "paper_v5" else V4_TASKS)
        self.pit_scene = 2 if "scene2" in launch_name else 1
        self.pit_centres = (() if self.fixed_pit_arena else
                            PIT_CENTRES[self.pit_scene] if self.with_pits else ())
        self.pit_width = (1.6 if self.arena_id == "paper_v6" else
                          1.8 if self.fixed_pit_arena else PIT_WIDTH)
        self.safety_stop_distance = float(safety_stop_distance)
        self.safety_slow_distance = float(safety_slow_distance)
        self.pass_turn_damping = float(pass_turn_damping)
        self.pass_usage_cost = float(pass_usage_cost)
        if self.use_pass_gate and not self.use_safety_shield:
            raise ValueError("pass gate requires safety shield")
        if self.safety_slow_distance <= self.safety_stop_distance:
            raise ValueError("safety_slow_distance must exceed safety_stop_distance")

        self.lidar_noise_std = float(lidar_noise_std)
        self.height_noise_std = float(height_noise_std)
        self.lidar_dropout_prob = float(lidar_dropout_prob)
        self.camera_noise_std = float(camera_noise_std)
        self.base_linear_response = float(linear_response)
        self.base_angular_response = float(angular_response)
        self.domain_randomization = bool(domain_randomization)
        self.linear_response = self.base_linear_response
        self.angular_response = self.base_angular_response
        self.seed(seed)

        self.odom_x = self.odom_y = self.robot_yaw = self.robot_z = 0.0
        self.robot_roll = self.robot_pitch = 0.0
        # Third row of the odometry quaternion rotation matrix. A tilted
        # robot sees level ground at a negative sensor-frame z; compensate
        # roll/pitch before interpreting negative heights as a pit.
        self.odom_vertical_row = (0.0, 0.0, 1.0)
        self.sensor_x = self.sensor_y = 0.0
        self.pit_front_latch = None
        self.goal_x, self.goal_y, self.prev_distance = 1.0, 0.0, 0.0
        self.curriculum_step = 0
        self.training_step = None  # Set only by the trainer, never by evaluation.
        self.current_task_type = "uninitialized"
        self.velodyne_data = np.full(self.environment_dim, 10.0, dtype=np.float32)
        self.obstacle_heights = np.zeros(self.environment_dim, dtype=np.float32)
        self.pit_memory = PitMemory(self.environment_dim) if self.with_pits else None
        self.last_odom = self._last_rgb = None
        self.last_pass_signal, self.last_pass_gate, self.last_pass_mode = -1.0, 0.0, 0.0
        self.last_linear_x = self.last_angular_z = self.theta = 0.0
        self._recent_angular, self._recent_linear, self._recent_progress = [], [], []
        self._event_reset()

        self.set_self_state = ModelState()
        self.set_self_state.model_name = self.robot_name
        self.set_self_state.pose.orientation.w = 1.0
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim])
        self.gaps[-1][-1] += 0.03

        if not connect_only:
            subprocess.Popen(["roscore", "-p", ros_port])
            time.sleep(1.0)
        rospy.init_node("traversability_td3", anonymous=True)
        if not connect_only:
            fullpath = self._resolve_launchfile(launchfile)
            if not os.path.exists(fullpath):
                raise IOError("Launch file does not exist: " + fullpath)
            subprocess.Popen(["roslaunch", "-p", ros_port, fullpath])

        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.set_model_state_service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.get_model_state_service = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher("pass_signal", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.model_states = rospy.Subscriber("/gazebo/model_states", ModelStates,
                                             self.model_states_callback, queue_size=1)
        self.rgb = rospy.Subscriber(camera_topic, Image, self.rgb_callback, queue_size=1) if self.use_rgb else None
        if startup_wait_s > 0:
            time.sleep(float(startup_wait_s))
        if require_topics:
            self._wait_for_ready(camera_topic)

    @property
    def observation_dim(self):
        return (self.environment_dim + self.robot_dim +
                (self.environment_dim if self.use_height else 0) +
                (2 if self.use_pass_gate else 0) +
                (10 if self.use_enhanced_features else 0))

    def seed(self, seed: int):
        self.random_seed = int(seed)
        self.scenario_random = np.random.RandomState(self.random_seed)
        self.noise_random = np.random.RandomState(self.random_seed + 1_000_003)
        self.dynamics_random = np.random.RandomState(self.random_seed + 2_000_003)
        self.np_random = self.scenario_random  # compatibility alias

    def get_rng_state(self):
        return (
            self.scenario_random.get_state(), self.noise_random.get_state(),
            self.dynamics_random.get_state(),
        )

    def set_rng_state(self, state):
        self.scenario_random.set_state(state[0])
        self.noise_random.set_state(state[1])
        self.dynamics_random.set_state(state[2])

    @staticmethod
    def _resolve_launchfile(launchfile: str) -> str:
        if not launchfile:
            raise ValueError("launchfile must be non-empty")
        if os.path.isabs(launchfile):
            return launchfile
        try:
            pkg = subprocess.check_output(["rospack", "find", "multi_robot_scenario"], text=True).strip()
            return os.path.join(pkg, "launch", launchfile)
        except Exception:
            return os.path.join(os.path.dirname(__file__), "assets", launchfile)

    def _wait_for_ready(self, camera_topic):
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")
        rospy.wait_for_service("/gazebo/reset_world")
        # The previous training subprocess leaves the shared Gazebo physics
        # paused after its final step. A newly created environment must resume
        # simulation before waiting for fresh odometry/point-cloud messages.
        try:
            self.unpause()
        except rospy.ServiceException as exc:
            raise RuntimeError("Gazebo is ready but could not unpause physics") from exc
        try:
            rospy.wait_for_message(self.odom_topic, Odometry, timeout=30.0)
            rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=30.0)
            if self.with_pits:
                rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=30.0)
            if self.use_rgb:
                rospy.wait_for_message(camera_topic, Image, timeout=30.0)
        finally:
            try:
                self.pause()
            except rospy.ServiceException:
                pass

    def velodyne_callback(self, cloud):
        distances = np.full(self.environment_dim, 10.0, dtype=np.float32)
        original_distances = np.full(self.environment_dim, 10.0, dtype=np.float32)
        heights = np.zeros(self.environment_dim, dtype=np.float32)
        sectors = [[] for _ in range(self.environment_dim)]
        floor_samples = []
        vertical = self.odom_vertical_row
        for x, y, z in pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z")):
            r_xy = math.hypot(x, y)
            if r_xy < 1e-6:
                continue
            if getattr(self, "use_original_observation", False):
                # Same height cutoff, angular bins and 3-D range as the author.
                # Clamp acos for roundoff and skip invalid/zero-range returns.
                update_original_sector(original_distances, self.gaps, x, y, z)
            point_height = float(
                vertical[0] * (x + SENSOR_X_OFFSET) + vertical[1] * y
                + vertical[2] * (z + SENSOR_HEIGHT)
            )
            if self.with_pits and 0.35 < r_xy < 3.0 and -0.18 <= point_height <= 0.18:
                floor_samples.append(point_height)
            beta = math.atan2(y, x)
            for sector, (lo, hi) in enumerate(self.gaps):
                if lo <= beta < hi:
                    sectors[sector].append((r_xy, point_height))
                    break
        if getattr(self, "use_original_observation", False):
            self.original_velodyne_data = original_distances
        floor_reference = float(np.median(floor_samples)) if floor_samples else 0.0
        if self.with_pits:
            distances, heights = signed_sector_observation(
                sectors, floor_reference, ground_tol=GROUND_TOL,
                pit_depth_tol=PIT_DEPTH_TOL,
                near_edge_offset=self.pit_width,
                project_to_ground=self.fixed_pit_arena,
                sensor_height=SENSOR_HEIGHT,
                associate_nearest=self.fixed_pit_arena,
            )
            front = range(max(0, self.environment_dim//2-2),
                          min(self.environment_dim, self.environment_dim//2+2))
            raw_pit = [i for i in front if heights[i] < -PIT_DEPTH_TOL and distances[i] < 2.0]
            if raw_pit:
                i = min(raw_pit, key=lambda j: distances[j])
                self.pit_front_latch = (float(distances[i]), self.sensor_x,
                                        self.sensor_y, self.robot_yaw,
                                        float(heights[i]))
            self.velodyne_data, self.obstacle_heights = self.pit_memory.update(
                distances, heights, self.sensor_x, self.sensor_y, self.robot_yaw,
            )
            if self.pit_front_latch is not None:
                r0, x0, y0, yaw0, pit_height = self.pit_front_latch
                turn = abs((self.robot_yaw-yaw0+math.pi) % (2*math.pi)-math.pi)
                lateral = abs(-(self.sensor_x-x0)*math.sin(yaw0)
                              +(self.sensor_y-y0)*math.cos(yaw0))
                advance = ((self.sensor_x-x0)*math.cos(yaw0)
                           +(self.sensor_y-y0)*math.sin(yaw0))
                if turn > 0.5 or lateral > 0.55 or advance > r0 + self.pit_width:
                    self.pit_front_latch = None
                else:
                    predicted = max(0.0, r0-advance)
                    centre = self.environment_dim//2
                    if predicted < self.velodyne_data[centre]:
                        self.velodyne_data[centre] = predicted
                        self.obstacle_heights[centre] = pit_height
            return
        for sector, points in enumerate(sectors):
            if not points:
                continue
            values = np.asarray(points, dtype=np.float32)
            ground = min(0.0, float(np.percentile(values[:, 1], 10)))
            above = values[:, 1] - ground
            mask = above > GROUND_TOL
            if np.any(mask):
                distances[sector] = float(np.min(values[mask, 0]))
                heights[sector] = float(np.max(above[mask]))
        self.velodyne_data, self.obstacle_heights = distances, heights

    def rgb_callback(self, msg: Image):
        try:
            from cv_bridge import CvBridge  # type: ignore
            rgb = CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            return
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] >= 3:
            rgb = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8)
            self._last_rgb = _resize_rgb(rgb, self.image_size_hw)

    def odom_callback(self, data):
        self.last_odom = data
        self.sensor_x = float(data.pose.pose.position.x)
        self.sensor_y = float(data.pose.pose.position.y)
        q = data.pose.pose.orientation
        self.odom_vertical_row = vertical_row_from_quaternion(q.w, q.x, q.y, q.z)
        self.robot_roll, self.robot_pitch, self.robot_yaw = (
            Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)
        )

    def model_states_callback(self, data):
        try:
            index = data.name.index(self.robot_name)
        except ValueError:
            return
        self.robot_z = float(data.pose[index].position.z)

    def _get_rgb_obs(self):
        h, w = self.image_size_hw
        if not self.use_rgb or self._last_rgb is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        rgb = self._last_rgb
        if rgb.shape[:2] != (h, w):
            rgb = _resize_rgb(rgb, self.image_size_hw)
        if self.camera_noise_std > 0:
            rgb = np.clip(rgb.astype(np.float32) + self.noise_random.normal(0, self.camera_noise_std, rgb.shape), 0, 255).astype(np.uint8)
        return rgb.copy()

    def _sensor_observation(self):
        distance = self.velodyne_data.astype(np.float32).copy()
        height = self.obstacle_heights.astype(np.float32).copy()
        if self.lidar_noise_std > 0:
            distance += self.noise_random.normal(0, self.lidar_noise_std, distance.shape).astype(np.float32)
        if self.height_noise_std > 0:
            height += self.noise_random.normal(0, self.height_noise_std, height.shape).astype(np.float32)
        if self.lidar_dropout_prob > 0:
            dropped = self.noise_random.rand(self.environment_dim) < self.lidar_dropout_prob
            distance[dropped], height[dropped] = 10.0, 0.0
        return np.clip(distance, 0, 10), np.clip(height, -1 if self.with_pits else 0, 1)

    def _front_obstacle(self, distance=None, height=None):
        distance = self.velodyne_data if distance is None else distance
        height = self.obstacle_heights if height is None else height
        mid = self.environment_dim // 2
        front = np.arange(max(0, mid - 2), min(self.environment_dim, mid + 2))
        # A nearer ordinary return must not hide a pit in an adjacent forward
        # sector; the safety gate needs the negative obstacle class first.
        pit_front = front[(height[front] < -PIT_DEPTH_TOL) & (distance[front] < 2.0)] if self.with_pits else []
        candidates = pit_front if len(pit_front) else front
        idx = int(candidates[int(np.argmin(distance[candidates]))])
        dist, obstacle_height = float(distance[idx]), float(height[idx])
        if dist > 2.0:
            return dist, obstacle_height, "none"
        if self.with_pits and obstacle_height < -PIT_DEPTH_TOL:
            return dist, obstacle_height, "pit"
        return dist, obstacle_height, "low" if obstacle_height < self.passable_height else "high"

    def _shield_scale(self, distance):
        if not self.use_safety_shield:
            return 1.0
        span = self.safety_slow_distance - self.safety_stop_distance
        return float(np.clip((distance - self.safety_stop_distance) / span, 0, 1))

    def _compute_enhanced_features(self, distance, height):
        idx = int(np.argmin(distance))
        mask = distance < FREE_DIST
        valid_height = height[mask]
        passable = mask & (height >= 0.0) & (height < self.passable_height)
        mid = self.environment_dim // 2
        is_low = 0.0 <= height[idx] < self.passable_height
        danger = (1 - np.clip(float(distance[idx]) / 2, 0, 1)) * (0.3 if is_low else 1.0)
        return np.asarray([
            height[idx], valid_height.mean() if valid_height.size else 0,
            valid_height.max() if valid_height.size else 0,
            float(passable.sum()) / max(int(mask.sum()), 1), float(idx) / self.environment_dim,
            danger, np.mean(distance[mid-2:mid+2]), np.mean(distance[:3]),
            np.mean(distance[-3:]), np.cos(self.theta),
        ], dtype=np.float32)

    def _build_state(self, goal_distance, theta, linear, angular, progress):
        if getattr(self, "use_original_observation", False):
            distance = self.original_velodyne_data.copy()
            if self.lidar_noise_std > 0:
                distance += self.noise_random.normal(0, self.lidar_noise_std, distance.shape)
            if self.lidar_dropout_prob > 0:
                distance[self.noise_random.rand(self.environment_dim) < self.lidar_dropout_prob] = 10.0
            return np.concatenate([np.clip(distance, 0, 10),
                                   [goal_distance, theta, linear, angular]]).astype(np.float32)
        distance, height = self._sensor_observation()
        pieces = [distance]
        if self.use_height:
            pieces.append(height)
        robot = [goal_distance, theta, linear, angular]
        if self.robot_dim == 5:
            robot.append(progress)
        pieces.append(np.asarray(robot, dtype=np.float32))
        if self.use_pass_gate:
            pieces.append(np.asarray([self.last_pass_signal, self.last_pass_gate], dtype=np.float32))
        if self.use_enhanced_features:
            pieces.append(self._compute_enhanced_features(distance, height))
        state = np.concatenate(pieces).astype(np.float32)
        if state.size != self.observation_dim:
            raise RuntimeError("observation size %d != configured %d" % (state.size, self.observation_dim))
        return state

    def observe_collision(self, laser_data):
        minimum = float(np.min(laser_data))
        collision = bool(np.any(
            (np.asarray(laser_data) < COLLISION_DIST)
            & (self.obstacle_heights >= self.passable_height)
        ))
        return bool(collision), bool(collision), minimum

    def get_reward(self, target, collision, action, min_laser, progress, theta, gate, pit_fall=False):
        if self.reward_mode == "ral2022":
            if target:
                return 100.0
            if collision or pit_fall:
                return -100.0
            if getattr(self, "use_original_observation", False):
                min_laser = float(np.min(self.original_velodyne_data))
            proximity = 1.0 - min_laser if min_laser < 1.0 else 0.0
            return float(action[0] / 2.0 - abs(action[1]) / 2.0 - proximity / 2.0)
        return shaped_reward(target, collision, pit_fall, action[0], action[1],
                             min_laser, progress, theta, gate, self.pass_usage_cost)

    def _event_reset(self):
        self._low_event_active, self._high_event_active, self._low_event_progress = False, False, 0.0

    def _event_update(self, before_class, before_distance, gate, travelled, collision):
        low_started = before_class == "low" and before_distance < 0.80 and gate >= 0.5 and not self._low_event_active
        high_started = before_class == "high" and before_distance < 0.80 and not self._high_event_active
        if low_started:
            self._low_event_active, self._low_event_progress = True, 0.0
        if high_started:
            self._high_event_active = True
        if self._low_event_active:
            self._low_event_progress += max(float(travelled), 0.0)
        after_distance, _, after_class = self._front_obstacle()
        low_success = self._low_event_active and self._low_event_progress > 0.15 and (after_class != "low" or after_distance > 1.0)
        high_success = self._high_event_active and not collision and (after_class != "high" or after_distance > 1.0)
        if low_success or collision:
            self._low_event_active = False
        if high_success or collision:
            self._high_event_active = False
        return low_started, low_success, high_started, high_success

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _pit_nearby(self):
        if getattr(self, 'evaluation_launchfile', None):
            return near_evaluation_pit(self.odom_x, self.odom_y, self.evaluation_launchfile)
        if self.arena_id == "paper_v6":
            return near_v6_pit(self.odom_x, self.odom_y)
        if self.arena_id == "paper_v5":
            return near_v5_pit(self.odom_x, self.odom_y)
        if self.arena_id == "paper_v4":
            return near_v4_pit(self.odom_x, self.odom_y)
        for x, y in self.pit_centres:
            dx = max(abs(self.odom_x - x) - PIT_WIDTH / 2, 0.0)
            dy = max(abs(self.odom_y - y) - PIT_WIDTH / 2, 0.0)
            if math.hypot(dx, dy) < 1.0:
                return True
        return False

    def _valid_fixed_position(self, x, y):
        if getattr(self, 'evaluation_launchfile', None):
            return valid_evaluation_position(x, y, self.evaluation_launchfile)
        if self.arena_id == "paper_v6":
            return valid_v6_position(x, y)
        return valid_v5_position(x, y) if self.arena_id == "paper_v5" else valid_v4_position(x, y)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 2:
            raise ValueError("action needs [linear, angular]")
        requested_linear = float(np.clip(action[0], 0, 1))
        requested_angular = float(np.clip(action[1], -1, 1))
        signal = float(np.clip(action[2], -1, 1)) if self.use_pass_gate else -1.0
        requested_gate = 0.5 * (signal + 1) if self.use_pass_gate else 0.0
        observed_distance, observed_height = self._sensor_observation()
        front_distance, front_height, front_class = self._front_obstacle(observed_distance, observed_height)
        # The third action is a learned *low-obstacle traversal* decision.
        # High barriers and pits cannot be overridden by that decision.
        gate = authorized_pass_gate(requested_gate, front_class)
        shield_scale = self._shield_scale(front_distance)
        guarded = requested_linear * shield_scale
        linear = ((1 - gate) * guarded + gate * requested_linear) * self.linear_response
        angular = requested_angular * (1 - gate * self.pass_turn_damping) * self.angular_response
        linear, angular = float(np.clip(linear, 0, 1)), float(np.clip(angular, -1, 1))
        self.last_pass_signal, self.last_pass_gate, self.last_pass_mode = signal, gate, float(gate >= 0.5)

        before_x, before_y = self.odom_x, self.odom_y
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = linear, angular
        self.vel_pub.publish(cmd)
        self.publish_markers([linear, angular, signal])
        self._run_physics_step()
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        pit_fall = bool(self.with_pits and self.robot_z < PIT_FALL_Z)
        # Report one terminal failure type per episode. A bottom/wall return
        # while the robot is already below grade is a fall, not a second hit.
        collision = bool(collision and not pit_fall)
        if self.last_odom is not None:
            self.odom_x, self.odom_y = self.last_odom.pose.pose.position.x, self.last_odom.pose.pose.position.y
            q = self.last_odom.pose.pose.orientation
            self.robot_yaw = Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
        distance = float(math.hypot(self.odom_x - self.goal_x, self.odom_y - self.goal_y))
        progress, self.prev_distance = self.prev_distance - distance, distance
        theta = self._wrap_angle(math.atan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x) - self.robot_yaw)
        target, done = distance < GOAL_REACHED_DIST, bool(done or pit_fall or distance < GOAL_REACHED_DIST)
        self.last_linear_x, self.last_angular_z, self.theta = linear, angular, theta

        self._recent_angular = (self._recent_angular + [abs(angular)])[-10:]
        self._recent_linear = (self._recent_linear + [linear])[-10:]
        self._recent_progress = (self._recent_progress + [progress])[-10:]
        spin_penalty = 0.0
        if len(self._recent_angular) == 10 and np.mean(self._recent_angular) > 0.45 and np.mean(self._recent_linear) < 0.12 and np.mean(self._recent_progress) < 0.005:
            spin_penalty = -0.15
        reward = self.get_reward(target, collision, [linear, angular], min_laser, progress, theta, gate, pit_fall)
        if self.reward_mode == "shaped":
            reward -= unsafe_pass_request_cost(requested_gate, front_class)
        if self.reward_mode == "ral2022" or target or collision or pit_fall:
            reward = float(reward)
        else:
            reward = float(np.clip(reward + spin_penalty, -1, 1))
        step_distance = float(math.hypot(self.odom_x-before_x, self.odom_y-before_y))
        low_start, low_ok, high_start, high_ok = self._event_update(front_class, front_distance, gate, step_distance, collision)
        info = {
            "target": bool(target), "collision": bool(collision),
            "pit_fall": pit_fall, "pit_detected": front_class == "pit",
            "pit_nearby": self._pit_nearby(),
            "distance": distance,
            "step_distance": step_distance,
            "min_laser": min_laser, "front_class": front_class,
            "front_distance": front_distance, "front_height": front_height,
            "pass_signal": signal,
            "pit_gate_suppressed": front_class == "pit" and requested_gate >= 0.5,
            "requested_gate": requested_gate,
            "gate_vetoed": front_class in ("high", "pit") and requested_gate >= 0.5,
            "pass_gate": gate, "gate_active": bool(gate >= 0.5),
            "shield_scale": shield_scale, "low_pass_started": low_start,
            "low_pass_success": low_ok, "high_avoid_started": high_start,
            "high_avoid_success": high_ok, "executed_linear": linear,
            "executed_angular": angular,
            "requested_linear": requested_linear,
            "requested_angular": requested_angular,
            "robot_x": float(self.odom_x), "robot_y": float(self.odom_y),
            "robot_yaw": float(self.robot_yaw),
            "robot_roll": float(self.robot_roll), "robot_pitch": float(self.robot_pitch),
            "goal_x": float(self.goal_x), "goal_y": float(self.goal_y),
            "task_type": self.current_task_type,
        }
        state = self._build_state(distance, theta, linear, angular, progress)
        return state, self._get_rgb_obs(), reward, done, info

    def _run_physics_step(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            pass

    def _place_robot_verified(self, state, target_x, target_y):
        """Teleport through a synchronous Gazebo service and verify the result.

        Publishing one ModelState topic message just after reset_world can race
        with Gazebo's reset and silently leave the robot at the world origin.
        A wrong start invalidates the entire fixed benchmark, so fail closed.
        """
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
        rospy.wait_for_service("/gazebo/get_model_state", timeout=10.0)
        for _ in range(3):
            applied = self.set_model_state_service(state)
            if not applied.success:
                raise RuntimeError("Gazebo rejected robot teleport: " + applied.status_message)
            self._run_physics_step()
            actual = self.get_model_state_service(self.robot_name, "world")
            if not actual.success:
                raise RuntimeError("Gazebo could not read robot pose: " + actual.status_message)
            px, py = float(actual.pose.position.x), float(actual.pose.position.y)
            if math.hypot(px - target_x, py - target_y) <= 0.15:
                self.odom_x, self.odom_y = px, py
                self.sensor_x, self.sensor_y = px, py
                q = actual.pose.orientation
                self.robot_roll, self.robot_pitch, self.robot_yaw = (
                    Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)
                )
                self.odom_vertical_row = vertical_row_from_quaternion(q.w, q.x, q.y, q.z)
                return
        raise RuntimeError(
            "Gazebo robot teleport did not reach (%.3f, %.3f); last pose (%.3f, %.3f)"
            % (target_x, target_y, px, py)
        )

    def reset(self, scenario=None):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            pass
        task_type = "benchmark" if scenario else "random"
        if scenario:
            episode_seed = int(scenario.get("noise_seed", self.random_seed))
            self.noise_random = np.random.RandomState(episode_seed)
            self.dynamics_random = np.random.RandomState(episode_seed + 1_000_003)
        if (scenario is None and self.arena_id in ("paper_v5", "paper_v6")
                and self.training_step is not None
                and self.scenario_random.rand() < open_task_probability(self.training_step)):
            start, goal = sample_v5_open_task(self.scenario_random, self.arena_id)
            scenario = {"robot_pose": start, "goal": goal}
            task_type = "open"
        pit_target = None
        if scenario:
            angle = float(scenario["robot_pose"][2])
        elif self.fixed_pit_arena and self.scenario_random.rand() < 0.30:
            task = self.pit_tasks[int(self.scenario_random.randint(len(self.pit_tasks)))]
            pit_target = task
            task_type = "pit"
            angle = task[0][2] + float(self.scenario_random.uniform(-0.15, 0.15))
        elif self.with_pits and not self.fixed_pit_arena and self.scenario_random.rand() < 0.30:
            px, py = self.pit_centres[int(self.scenario_random.randint(len(self.pit_centres)))]
            pit_target = (px, py + float(self.scenario_random.uniform(-0.08, 0.08)))
            task_type = "pit"
            angle = float(self.scenario_random.uniform(-0.20, 0.20))
        else:
            angle = float(self.scenario_random.uniform(-math.pi, math.pi))
        q = Quaternion.from_euler(0, 0, angle)
        if scenario:
            x, y = map(float, scenario["robot_pose"][:2])
        elif self.fixed_pit_arena and pit_target is not None:
            x, y = pit_target[0][:2]
        elif pit_target is not None:
            x, y = pit_target[0] - 1.75, pit_target[1]
        else:
            while True:
                x, y = float(self.scenario_random.uniform(-6.6, 6.6)), float(self.scenario_random.uniform(-5.6, 5.6))
                if (self._valid_fixed_position(x, y) if self.fixed_pit_arena else
                        is_safe_spawn_position(x, y, self.with_pits)):
                    break
        state = self.set_self_state
        state.pose.position.x, state.pose.position.y = x, y
        state.pose.position.z = 0.0
        state.pose.orientation.x, state.pose.orientation.y = q.x, q.y
        state.pose.orientation.z, state.pose.orientation.w = q.z, q.w
        state.twist = Twist()
        state.reference_frame = "world"
        self.odom_x, self.odom_y, self.robot_yaw = x, y, angle
        self.robot_roll = self.robot_pitch = 0.0
        self.odom_vertical_row = (0.0, 0.0, 1.0)
        self.sensor_x, self.sensor_y = x, y
        self.robot_z = 0.0
        self.current_task_type = task_type
        self.pit_front_latch = None
        if self.pit_memory is not None:
            self.pit_memory.clear()
        self.last_pass_signal, self.last_pass_gate, self.last_pass_mode = -1.0, 0.0, 0.0
        self.last_linear_x = self.last_angular_z = 0.0
        self._recent_angular, self._recent_linear, self._recent_progress = [], [], []
        self._event_reset()
        if self.domain_randomization:
            self.linear_response = self.base_linear_response * float(self.dynamics_random.uniform(0.75, 1.10))
            self.angular_response = self.base_angular_response * float(self.dynamics_random.uniform(0.80, 1.20))
        else:
            self.linear_response, self.angular_response = self.base_linear_response, self.base_angular_response
        if scenario:
            self.goal_x, self.goal_y = map(float, scenario["goal"])
            if not self.fixed_pit_arena:
                self.random_box(scenario.get("boxes"))
        else:
            if self.fixed_pit_arena and pit_target is not None:
                self.goal_x, self.goal_y = pit_target[1]
            elif pit_target is not None:
                self.goal_x, self.goal_y = pit_target[0] + 1.75, pit_target[1]
            else:
                self.change_goal()
            if not self.fixed_pit_arena:
                self.random_box()
        self.publish_markers([0, 0, -1])
        self._place_robot_verified(state, x, y)
        distance = float(math.hypot(self.odom_x-self.goal_x, self.odom_y-self.goal_y))
        self.prev_distance = distance
        theta = self._wrap_angle(math.atan2(self.goal_y-self.odom_y, self.goal_x-self.odom_x) - self.robot_yaw)
        self.theta = theta
        return self._build_state(distance, theta, 0, 0, 0), self._get_rgb_obs()

    def change_goal(self):
        self.curriculum_step += 1
        while True:
            x, y = float(self.scenario_random.uniform(-6.6, 6.6)), float(self.scenario_random.uniform(-5.6, 5.6))
            valid = (self._valid_fixed_position(x, y) if self.fixed_pit_arena else
                     is_valid_goal_position(x, y, self.with_pits))
            if valid and math.hypot(x-self.odom_x, y-self.odom_y) >= 3.0:
                self.goal_x, self.goal_y = x, y
                return

    def random_box(self, positions=None):
        for i in range(4):
            if positions is not None:
                x, y = map(float, positions[i])
            else:
                while True:
                    x, y = float(self.scenario_random.uniform(-6.6, 6.6)), float(self.scenario_random.uniform(-5.6, 5.6))
                    valid = is_valid_dynamic_obstacle_position(x, y, self.with_pits)
                    valid &= math.hypot(x-self.odom_x, y-self.odom_y) >= 1.5
                    valid &= math.hypot(x-self.goal_x, y-self.goal_y) >= 1.5
                    if valid:
                        break
            box = ModelState()
            box.model_name = "cardboard_box_%d" % i
            box.pose.position.x, box.pose.position.y, box.pose.orientation.w = x, y, 1.0
            self.set_state.publish(box)

    def publish_markers(self, action):
        def bar(publisher, value, y, color):
            array, marker = MarkerArray(), Marker()
            marker.header.frame_id, marker.type, marker.action = "odom", Marker.CUBE, Marker.ADD
            marker.scale.x, marker.scale.y, marker.scale.z = max(abs(float(value)), .001), .1, .01
            marker.color.a = 1.0
            marker.color.r, marker.color.g, marker.color.b = color
            marker.pose.orientation.w, marker.pose.position.x, marker.pose.position.y = 1.0, 5.0, y
            array.markers.append(marker)
            publisher.publish(array)
        goals, goal = MarkerArray(), Marker()
        goal.header.frame_id, goal.type, goal.action = "odom", Marker.CYLINDER, Marker.ADD
        # Visual-only marker: large enough to see beside the 3D point cloud.
        goal.scale.x, goal.scale.y, goal.scale.z = .24, .24, .06
        goal.color.a, goal.color.g, goal.pose.orientation.w = 1.0, 1.0, 1.0
        goal.pose.position.x, goal.pose.position.y = self.goal_x, self.goal_y
        goal.pose.position.z = .03
        goals.markers.append(goal)
        self.publisher.publish(goals)
        bar(self.publisher2, action[0], 0.0, (1, 0, 0))
        bar(self.publisher3, action[1], 0.2, (0, 0, 1))
        bar(self.publisher4, action[2], 0.4, (1, .8, 0))
