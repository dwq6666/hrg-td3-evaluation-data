"""Generate/load fixed benchmark episodes shared by every comparison method."""

import json
import math
import os

import numpy as np
from arena_config import (V4_TASKS, V5_TASKS, V6_TASKS, valid_v4_position,
                          valid_v5_position, valid_v6_position,
                          valid_evaluation_position, is_evaluation_only, world_sha256,
                          layout_pit_tasks)


HAZARDS = (
    (-6.15, -3.05, 2.60, 5.50), (-2.45, 0.45, 2.60, 5.50),
    (0.55, 3.65, 2.55, 5.55), (3.35, 6.35, 2.55, 5.55),
    (-6.45, -3.70, -1.55, 1.55), (3.25, 6.25, -1.55, 1.55),
    (-6.10, -3.00, -5.00, -2.20), (-1.95, 1.15, -5.00, -2.15),
    (2.90, 6.00, -5.00, -2.10),
)
SPECIAL = ()
PIT_MASKS = tuple(
    (x - 0.85, x + 0.85, y - 0.85, y + 0.85)
    for x in (-2.2, 2.2) for y in (-1.75, 1.75)
)


def inside(x, y, region):
    return region[0] <= x <= region[1] and region[2] <= y <= region[3]


def valid_goal(x, y, with_pits=False):
    return (abs(x) <= 6.6 and abs(y) <= 5.6
            and not any(inside(x, y, r) for r in HAZARDS)
            and not (with_pits and any(inside(x, y, r) for r in PIT_MASKS)))


def valid_spawn(x, y, with_pits=False):
    return valid_goal(x, y, with_pits) and not any(inside(x, y, r) for r in SPECIAL)


def valid_box(x, y, with_pits=False):
    return valid_spawn(x, y, with_pits) and not inside(x, y, (-1.0, 1.0, -1.0, 1.0))


def valid_fixed_position(x, y, arena, launchfile=None):
    if launchfile and is_evaluation_only(launchfile):
        return valid_evaluation_position(x, y, launchfile)
    if arena == "paper_v6":
        return valid_v6_position(x, y)
    return valid_v5_position(x, y) if arena == "paper_v5" else valid_v4_position(x, y)


def generate_scenarios(count=100, seed=100000, with_pits=False, arena="paper_v2", launchfile=None):
    rng = np.random.RandomState(int(seed))
    scenarios = []
    for episode in range(int(count)):
        fixed_pit_case = arena in ("paper_v4", "paper_v5", "paper_v6") and episode % 5 < 2
        if fixed_pit_case:
            tasks = (layout_pit_tasks(launchfile) if launchfile and is_evaluation_only(launchfile) else
                     V6_TASKS if arena == "paper_v6" else
                     V5_TASKS if arena == "paper_v5" else V4_TASKS)
            start, goal = tasks[episode % len(tasks)]
            x, y, yaw = start
            gx, gy = goal
            yaw += float(rng.uniform(-0.15, 0.15))
        elif with_pits and arena not in ("paper_v4", "paper_v5", "paper_v6") and episode < 40:
            quadrant = episode // 10
            cx = -2.2 if quadrant < 2 else 2.2
            cy = 1.75 if quadrant % 2 == 0 else -1.75
            y = gy = cy + float(rng.uniform(-0.08, 0.08))
            x, gx = cx - 1.75, cx + 1.75
            yaw = float(rng.uniform(-0.20, 0.20))
            if not valid_spawn(x, y, True) or not valid_goal(gx, gy, True):
                raise RuntimeError("pit benchmark crossing endpoints overlap a hazard mask")
        else:
            while True:
                x, y = float(rng.uniform(-6.6, 6.6)), float(rng.uniform(-5.6, 5.6))
                if (valid_fixed_position(x, y, arena, launchfile)
                        if arena in ("paper_v4", "paper_v5", "paper_v6") else
                        valid_spawn(x, y, with_pits)):
                    break
            yaw = float(rng.uniform(-math.pi, math.pi))
            while True:
                gx, gy = float(rng.uniform(-6.6, 6.6)), float(rng.uniform(-5.6, 5.6))
                goal_ok = (valid_fixed_position(gx, gy, arena, launchfile)
                           if arena in ("paper_v4", "paper_v5", "paper_v6") else
                           valid_goal(gx, gy, with_pits))
                if goal_ok and math.hypot(gx - x, gy - y) >= 3.0:
                    break
        boxes = []
        while arena not in ("paper_v4", "paper_v5", "paper_v6") and len(boxes) < 4:
            bx, by = float(rng.uniform(-6.6, 6.6)), float(rng.uniform(-5.6, 5.6))
            separated = all(math.hypot(bx-p[0], by-p[1]) >= 0.8 for p in boxes)
            if valid_box(bx, by, with_pits) and separated and math.hypot(bx-x, by-y) >= 1.5 and math.hypot(bx-gx, by-gy) >= 1.5:
                boxes.append([bx, by])
        scenarios.append({
            "episode": episode, "robot_pose": [x, y, yaw],
            "goal": [gx, gy], "boxes": boxes,
            "case_type": ("pit_crossing_candidate" if fixed_pit_case
                          else "pit_crossing_candidate" if with_pits and arena not in ("paper_v4", "paper_v5", "paper_v6") and episode < 40
                          else "random"),
            "noise_seed": int(seed) + episode * 9973,
        })
    return scenarios


def load_or_create(filepath, count=100, seed=100000, with_pits=False, arena="paper_v2", launchfile=None):
    filepath = os.path.abspath(filepath)
    version = 6 if arena == "paper_v6" else 5 if arena == "paper_v5" else 4 if arena == "paper_v4" else 3 if with_pits else 2
    heldout = os.path.basename(launchfile) if launchfile and is_evaluation_only(launchfile) else None
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (payload.get("version") != version or payload.get("seed") != int(seed)
                or payload.get('evaluation_layout') != heldout
                or (heldout and payload.get('world_sha256') != world_sha256(heldout))
                or (arena in ("paper_v4", "paper_v5", "paper_v6") and payload.get("layout") != "interleaved_pit_40pct")):
            raise ValueError(
                "benchmark manifest version/seed does not match selected arena; "
                "use a separate pit-v3 manifest and run root"
            )
        scenarios = payload["scenarios"]
        if len(scenarios) < count:
            raise ValueError("scenario manifest contains %d episodes, need %d" % (len(scenarios), count))
        for scenario in scenarios[:count]:
            x, y = scenario["robot_pose"][:2]
            gx, gy = scenario["goal"]
            valid_start = (valid_fixed_position(x, y, arena, launchfile)
                           if arena in ("paper_v4", "paper_v5", "paper_v6") else
                           valid_spawn(x, y, with_pits))
            valid_end = (valid_fixed_position(gx, gy, arena, launchfile)
                         if arena in ("paper_v4", "paper_v5", "paper_v6") else
                         valid_goal(gx, gy, with_pits))
            if not valid_start or not valid_end:
                raise ValueError("benchmark manifest contains a pose inside an obstacle or pit mask")
        return scenarios[:count]
    scenarios = generate_scenarios(count if heldout else max(100, count) if with_pits else count,
                                   seed, with_pits, arena, launchfile)
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump({"version": version, "seed": int(seed),
                   "evaluation_layout": heldout,
                   "world_sha256": world_sha256(heldout) if heldout else None,
                   "layout": "interleaved_pit_40pct" if arena in ("paper_v4", "paper_v5", "paper_v6") else "legacy",
                   "scenarios": scenarios}, handle, indent=2)
    print("Created fixed benchmark: %s" % filepath)
    return scenarios[:count]
