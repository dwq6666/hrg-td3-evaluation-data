"""Shared, geometry-checked open-task curriculum (never used in evaluation)."""

import math

from arena_config import V5_OBSTACLES, V5_PITS, V6_OBSTACLES, V6_PITS, inside


def open_task_probability(step):
    """Fade in difficult navigation while retaining some easy goals."""
    step = max(0, int(step))
    if step < 20_000:
        return 0.60
    if step < 100_000:
        return 0.35
    return 0.10


def _clear(x, y, arena="paper_v5"):
    if not (-6.40 <= x <= 6.40 and -5.40 <= y <= 5.40):
        return False
    hazards = (V6_PITS + V6_OBSTACLES if arena == "paper_v6"
               else V5_PITS + V5_OBSTACLES)
    return not any(inside(x, y, box, 0.55) for box in hazards)


def sample_v5_open_task(rng, arena="paper_v5"):
    """Return a short start/goal pair with a conservative free straight path."""
    for _ in range(512):
        x, y = float(rng.uniform(-6.2, 6.2)), float(rng.uniform(-5.2, 5.2))
        if not _clear(x, y, arena):
            continue
        bearing = float(rng.uniform(-math.pi, math.pi))
        length = float(rng.uniform(1.0, 2.2))
        gx, gy = x + length * math.cos(bearing), y + length * math.sin(bearing)
        samples = max(2, int(math.ceil(length / 0.10)))
        if all(_clear(x + (gx - x) * i / samples,
                      y + (gy - y) * i / samples, arena)
               for i in range(samples + 1)):
            yaw = bearing + float(rng.uniform(-0.30, 0.30))
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            return (x, y, yaw), (gx, gy)
    # Known clear southern lane; reachable even if rejection is unlucky.
    return (0.0, -5.0, 0.0), (1.5, -5.0)


def open_task_is_clear(start, goal, arena="paper_v5"):
    """Diagnostic predicate used by tests and preflight tools."""
    x, y = start[:2]
    gx, gy = goal
    length = math.hypot(gx - x, gy - y)
    samples = max(2, int(math.ceil(length / 0.10)))
    return all(_clear(x + (gx - x) * i / samples,
                      y + (gy - y) * i / samples, arena)
               for i in range(samples + 1))
