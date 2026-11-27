"""Author-style laser sectors, with guards for invalid sensor returns."""

import math
import numpy as np


def update_sector(distances, gaps, x, y, z):
    if not all(math.isfinite(v) for v in (x, y, z)) or z <= -0.2:
        return
    horizontal = math.hypot(x, y)
    if horizontal < 1e-6:
        return
    beta = math.acos(float(np.clip(x / horizontal, -1, 1))) * np.sign(y)
    distance = math.sqrt(x*x + y*y + z*z)
    for sector, (lo, hi) in enumerate(gaps):
        if lo <= beta < hi:
            distances[sector] = min(distances[sector], distance)
            break
