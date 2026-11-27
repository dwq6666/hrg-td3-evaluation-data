"""Versioned geometry shared by training, evaluation and preflight checks.

The rectangles below are used only for safe reset/goal sampling and metrics.
They are never supplied to a policy or to the LiDAR pit detector.
"""

import hashlib
import json
import os
from functools import lru_cache


WORLD_SHA256_V4 = "d967fd95830efd1d186cfbf797c002b0104ece12345a1d23285de5ca6bb49fe8"
WORLD_SHA256_V5 = "83d04d3f8c44b230f277335f3626eba8d5a022916258b29be384aafd9ba55df0"
WORLD_SHA256_V5_SCENE2 = "64954827aa4ae1bfe95ba4a6120f79daa85b7bada11ac67f4cbe235fa3e36240"
# The previous V6 scene is retained as paper_traversability_v6_archive.world.
# The active scene supplied on 2026-10-01 removes nine visual-only station tiles.
WORLD_SHA256_V6_ARCHIVE = "6f6b0d3ff747c8e7c072b59ce803b2eb03b67e5cade8f72703be06884290b4b6"
WORLD_SHA256_V6 = "f838796ec159deba29932ab69d63aa8fc0ae4eb92ca2baded778e1e33e0bf5de"
WORLD_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "catkin_ws", "src",
    "multi_robot_scenario", "launch"))


@lru_cache(maxsize=1)
def _evaluation_layouts():
    assets = os.path.join(os.path.dirname(__file__), 'assets')
    with open(os.path.join(assets, 'v6_holdout_layouts.json'), encoding='utf-8') as handle:
        layouts = json.load(handle)['layouts']
    # Revised worlds have their own frozen manifest and launch names. Keep the
    # original registrations intact so prior results retain their provenance.
    revised = os.path.join(assets, 'v6_holdout_layouts_v2_20261030.json')
    if os.path.exists(revised):
        with open(revised, encoding='utf-8') as handle:
            additions = json.load(handle)['layouts']
        if set(layouts).intersection(additions):
            raise ValueError('Revised evaluation layouts must have distinct launch names')
        layouts.update(additions)
    return layouts


def evaluation_layout(launchfile):
    return _evaluation_layouts().get(os.path.basename(launchfile))


def is_evaluation_only(launchfile):
    return evaluation_layout(launchfile) is not None


def expected_training_world(launchfile):
    layout = evaluation_layout(launchfile)
    if layout:
        return layout['source_world_sha256']
    return WORLD_SHA256_V5 if is_v5_scene2(launchfile) else world_sha256(launchfile)


def validate_checkpoint_origin(metadata, launchfile):
    if (metadata.get('arena_id') != arena_id(launchfile)
            or metadata.get('world_sha256') != expected_training_world(launchfile)):
        raise ValueError('checkpoint training world is not an approved source for this evaluation')


def valid_evaluation_position(x, y, launchfile):
    layout = evaluation_layout(launchfile)
    if not layout:
        raise ValueError('Unknown evaluation-only layout')
    return (-6.55 <= x <= 6.55 and -5.55 <= y <= 5.55
            and not any(inside(x, y, rectangle, .45) for rectangle in
                        list(layout.get('pit_rectangles', V6_PITS))+layout['obstacles']))


def near_evaluation_pit(x, y, launchfile, margin=1.0):
    layout = evaluation_layout(launchfile)
    if not layout:
        raise ValueError('Unknown evaluation-only layout')
    return any(inside(x, y, rectangle, margin)
               for rectangle in layout.get('pit_rectangles', V6_PITS))


def layout_pit_tasks(launchfile):
    layout = evaluation_layout(launchfile)
    if not layout:
        raise ValueError('Unknown evaluation-only layout')
    tasks = layout.get('pit_tasks', V6_TASKS)
    if not tasks:
        raise ValueError('Evaluation layout has no pit benchmark tasks')
    return tasks

# (xmin, xmax, ymin, ymax): actual hole footprints, not the padded reset masks.
V4_PITS = ((-5.5, -3.7, 1.3, 2.9), (1.4, 3.8, 1.175, 1.825))
V4_HIGH = (
    (-5.35, -4.45, -1.05, -0.25),
    (-1.85, -0.65, 1.60, 2.30),
    (-0.25, 0.55, 3.45, 4.25),
    (4.60, 5.40, 2.30, 3.20),
    (4.85, 5.65, -1.00, -0.20),
    (-6.00, -5.30, 4.07, 4.83),
    (1.95, 2.75, 4.00, 4.80),
    (-1.02, -0.28, -4.99, -4.31),
)
V4_LOW = (
    (-3.11, -2.25, -3.08, -1.62),
    (-5.61, -4.79, -4.75, -3.55),
    (0.52, 1.58, -2.67, -2.04),
    (2.72, 4.83, -3.70, -2.40),
    (-1.84, -0.31, 4.05, 4.94),
)
V4_TASKS = (
    ((-6.25, 2.10, 0.0), (-2.85, 2.10)),
    ((2.60, 0.20, 1.5707963267948966), (2.60, 2.80)),
)

# The supplied Scene I replacement has a central pit and southern trench.
# Static exhibit zones and boxes are excluded from reset/goal sampling.
V5_PITS = ((-1.60, -0.20, 0.90, 2.25), (0.70, 2.70, -2.15, -1.50))
V5_OBSTACLES = (
    (-5.95, -3.35, 3.30, 5.00), (-1.80, -0.20, 3.55, 4.50),
    (0.85, 3.35, 3.50, 4.50), (3.60, 5.95, 3.15, 5.05),
    (-5.95, -4.05, -1.35, 1.35), (3.60, 5.80, -1.10, 1.10),
    (-5.90, -3.20, -4.60, -2.55), (-1.25, 0.55, -4.00, -3.05),
    (3.30, 5.75, -4.30, -2.50),
    (-3.13, -2.77, 0.47, 0.83), (-3.08, -2.72, -0.93, -0.57),
    (2.92, 3.28, 0.67, 1.03), (2.97, 3.33, -1.03, -0.67),
    (-5.45, -2.15, 2.72, 3.00), (1.38, 1.62, -0.855, 2.395),
    (4.675, 4.925, -3.885, -1.455),
)
V5_TASKS = (
    ((-0.90, -0.10, 1.5707963267948966), (-0.90, 2.95)),
    ((2.30, -0.50, -1.5707963267948966), (2.30, -3.00)),
)

# Supplied V6 Scene I: isolated circular pit west and circular trench east.
# These masks are for safe resets/benchmark metadata only, never policy input.
V6_PITS = ((-5.26, -3.74, 1.19, 2.71), (3.78, 5.22, -2.52, -1.08))
V6_OBSTACLES = (
    (-3.13, -2.77, 0.47, 0.83), (2.92, 3.28, 0.67, 1.03),
    (-5.28, -4.32, -0.25, 0.35), (-4.75, -3.85, -4.25, -3.35),
    (2.05, 3.85, 2.615, 2.745), (-1.305, -1.175, -1.62, -0.42),
    (-1.75, -0.30, 3.70, 4.46), (-1.20, -0.19, -3.89, -3.13),
    (0.85, 3.28, 3.64, 4.36), (3.75, 4.25, 3.2, 4.9),
    (-2.35, -1.45, 2.93, 3.07), (-3.47, -3.33, -2.75, -1.95),
    (1.65, 2.45, -3.47, -3.33),
)
V6_TASKS = (
    ((-6.00, 1.95, 0.0), (-3.00, 1.95)),
    ((2.70, -1.80, 0.0), (6.20, -1.80)),
)


def arena_id(launchfile):
    name = os.path.basename(launchfile)
    if name == "paper_traversability.launch" or is_evaluation_only(name):
        return "paper_v6"
    if name in ("paper_traversability_v5.launch", "paper_traversability_scene2.launch"):
        return "paper_v5"
    if name == "paper_traversability_v4.launch":
        return "paper_v4"
    if "pits" in name:
        return "paper_pits_v3"
    return "paper_v2"


def world_path(launchfile):
    return os.path.join(WORLD_DIR, os.path.splitext(os.path.basename(launchfile))[0] + ".world")


def world_sha256(launchfile):
    digest = hashlib.sha256()
    with open(world_path(launchfile), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_versioned_world(launchfile):
    layout = evaluation_layout(launchfile)
    expected = (layout['world_sha256'] if layout else {"paper_v4": WORLD_SHA256_V4,
                 "paper_v5": (WORLD_SHA256_V5_SCENE2 if is_v5_scene2(launchfile)
                              else WORLD_SHA256_V5),
                 "paper_v6": WORLD_SHA256_V6}.get(arena_id(launchfile)))
    if expected and world_sha256(launchfile) != expected:
        raise ValueError("world file differs from the approved %s arena; check deployment"
                         % arena_id(launchfile))


def is_v5_scene2(launchfile):
    return os.path.basename(launchfile) == "paper_traversability_scene2.launch"


def inside(x, y, rect, padding=0.0):
    return (rect[0] - padding <= x <= rect[1] + padding
            and rect[2] - padding <= y <= rect[3] + padding)


def valid_v4_position(x, y):
    if not (-6.55 <= x <= 6.55 and -5.55 <= y <= 5.55):
        return False
    return not any(inside(x, y, r, 0.45) for r in V4_PITS + V4_HIGH + V4_LOW)


def near_v4_pit(x, y, margin=1.0):
    return any(inside(x, y, r, margin) for r in V4_PITS)


def valid_v5_position(x, y):
    if not (-6.55 <= x <= 6.55 and -5.55 <= y <= 5.55):
        return False
    return not any(inside(x, y, r, 0.45) for r in V5_PITS + V5_OBSTACLES)


def near_v5_pit(x, y, margin=1.0):
    return any(inside(x, y, r, margin) for r in V5_PITS)


def valid_v6_position(x, y):
    if not (-6.55 <= x <= 6.55 and -5.55 <= y <= 5.55):
        return False
    return not any(inside(x, y, r, 0.45) for r in V6_PITS + V6_OBSTACLES)


def near_v6_pit(x, y, margin=1.0):
    return any(inside(x, y, r, margin) for r in V6_PITS)
