"""Pure geometry extraction for the physically recessed pit arenas.

The policy receives signed elevations: positive means a raised obstacle,
negative means a LiDAR return below the surrounding floor. No pit world
coordinate or ground-truth label enters this function.
"""

import math

import numpy as np


def vertical_row_from_quaternion(w, x, y, z):
    """World-up projection of a robot-frame vector using odometry attitude."""
    return (
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def signed_sector_observation(sectors, floor_reference, *,
                              ground_tol=0.01, pit_depth_tol=0.08,
                              near_edge_offset=0.90, project_to_ground=False,
                              sensor_height=0.25, edge_margin=0.25,
                              associate_nearest=False):
    distances = np.full(len(sectors), 10.0, dtype=np.float32)
    heights = np.zeros(len(sectors), dtype=np.float32)
    for sector, points in enumerate(sectors):
        if not points:
            continue
        values = np.asarray(points, dtype=np.float32)
        relative = values[:, 1] - float(floor_reference)
        positive = relative > ground_tol
        negative = relative < -pit_depth_tol
        positive_distance = float(np.min(values[positive, 0])) if np.any(positive) else 10.0
        if np.any(negative) and project_to_ground:
            # Project each downward ray to the nominal ground plane. This
            # estimates the first unsupported floor point without assuming a
            # fixed pit width (the v4 arena has a pit and a narrower trench).
            ranges = values[negative, 0]
            depths = -relative[negative]
            crossings = ranges * sensor_height / (sensor_height + depths)
            pit_distance = max(0.0, float(np.min(crossings)) - edge_margin)
        else:
            # Legacy v3 width-based approximation, kept for old experiments.
            pit_distance = (max(0.0, float(np.min(values[negative, 0])) - near_edge_offset)
                            if np.any(negative) else 10.0)
        if np.any(negative) and pit_distance < positive_distance:
            distances[sector] = pit_distance
            heights[sector] = float(np.min(relative[negative]))
        elif np.any(positive):
            distances[sector] = positive_distance
            local = positive & (values[:, 0] <= positive_distance + 0.15)
            heights[sector] = float(np.max(relative[local if associate_nearest else positive]))
    return distances, heights


class PitMemory:
    """Keep only LiDAR-observed pit edges across brief scan occlusions.

    The stored coordinates are inferred from range/yaw, never read from the
    Gazebo world. Memory is cleared at every episode reset.
    """

    def __init__(self, n_sectors, max_range=3.0):
        self.n = int(n_sectors)
        self.max_range = float(max_range)
        self.points = []

    def clear(self):
        self.points = []

    def update(self, distance, height, x, y, yaw, pit_threshold=-0.10):
        angles = -math.pi / 2 + (np.arange(self.n) + 0.5) * math.pi / self.n
        for index in np.flatnonzero(height < pit_threshold):
            r = float(distance[index])
            if r > self.max_range:
                continue
            world_angle = yaw + float(angles[index])
            px, py = x + r * math.cos(world_angle), y + r * math.sin(world_angle)
            replacement = next(
                (j for j, (mx, my, _) in enumerate(self.points)
                 if math.hypot(px-mx, py-my) < 0.25), None
            )
            item = (px, py, float(height[index]))
            if replacement is None:
                self.points.append(item)
            else:
                self.points[replacement] = item
        self.points = [(px, py, h) for px, py, h in self.points
                       if math.hypot(px-x, py-y) <= self.max_range]
        for px, py, h in self.points:
            dx, dy = px-x, py-y
            r = math.hypot(dx, dy)
            bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            if not (-math.pi/2 <= bearing < math.pi/2):
                continue
            index = min(self.n-1, max(0, int((bearing+math.pi/2) / math.pi * self.n)))
            if r < float(distance[index]):
                distance[index] = max(0.0, r)
                height[index] = h
        return distance, height
