"""Classical controller baselines evaluated through the same environment."""

import math
import json
import hashlib

import numpy as np


DEFAULT_PARAMETERS = dict(horizon=1.5, dt=.1, robot_radius=.35,
                          velocity_window=.15, turn_window=.30, velocity_samples=7,
                          turn_samples=15, clearance_margin=.05,
                          progress_weight=2., clearance_weight=.35,
                          heading_weight=.55, turn_weight=.08, pit_height=-.08)


def load_controller_parameters(path, name):
    if path is None:
        return dict(DEFAULT_PARAMETERS), None
    with open(path, 'rb') as handle:
        content = handle.read()
    payload = json.loads(content)
    parameters = payload['controllers'][name]
    if set(parameters) != set(DEFAULT_PARAMETERS):
        raise ValueError('Incomplete or unknown controller parameters')
    if any(not math.isfinite(float(v)) for v in parameters.values()):
        raise ValueError('Non-finite controller parameter')
    # Geometry/control limits are shared physical conditions, never search knobs.
    for key in ('robot_radius','dt','velocity_window','turn_window','velocity_samples',
                'turn_samples','clearance_margin','pit_height'):
        if parameters[key] != DEFAULT_PARAMETERS[key]:
            raise ValueError('Physical/control parameter changed: '+key)
    if parameters['horizon'] <= 0 or min(parameters[k] for k in
            ('progress_weight','clearance_weight','heading_weight','turn_weight')) < 0:
        raise ValueError('Invalid DWA scoring parameters')
    return parameters, hashlib.sha256(content).hexdigest()


class DynamicWindowController:
    """Local Dynamic Window Approach over sector-endpoint obstacle points."""

    def __init__(self, cfg, horizon=1.5, dt=0.1, robot_radius=0.35, parameters=None):
        self.cfg = cfg
        self.parameters = dict(DEFAULT_PARAMETERS)
        self.parameters.update(horizon=horizon,dt=dt,robot_radius=robot_radius)
        if parameters is not None:
            self.parameters.update(parameters)
        self.horizon = float(self.parameters['horizon'])
        self.dt = float(self.parameters['dt'])
        self.robot_radius = float(self.parameters['robot_radius'])

    @staticmethod
    def _wrap(value):
        return (value + math.pi) % (2 * math.pi) - math.pi

    def get_action(self, state, image=None):
        cfg, n = self.cfg, self.cfg.n_sectors
        distance = np.asarray(state[:n], dtype=np.float32)
        offset = n
        if cfg.use_height:
            height = np.asarray(state[offset:offset+n], dtype=np.float32)
            offset += n
        else:
            height = np.ones(n, dtype=np.float32)
        theta = float(state[offset+1])
        previous_v, previous_w = float(state[offset+2]), float(state[offset+3])
        angles = np.linspace(-math.pi/2, math.pi/2, n, endpoint=False) + math.pi/(2*n)
        obstacle = distance < 5.0
        if cfg.use_height:
            obstacle &= (height >= cfg.passable_height) | (height < self.parameters['pit_height'])
        points = np.stack([distance[obstacle]*np.cos(angles[obstacle]), distance[obstacle]*np.sin(angles[obstacle])], axis=1) if np.any(obstacle) else np.empty((0, 2))

        p = self.parameters
        v_min, v_max = max(0.0, previous_v-p['velocity_window']), min(1.0, previous_v+p['velocity_window'])
        w_min, w_max = max(-1.0, previous_w-p['turn_window']), min(1.0, previous_w+p['turn_window'])
        velocities, turns = np.linspace(v_min, v_max, int(p['velocity_samples'])), np.linspace(w_min, w_max, int(p['turn_samples']))
        best_score, best = -1e9, np.asarray([0.0, 0.0], dtype=np.float32)
        for velocity in velocities:
            for turn in turns:
                x = y = yaw = 0.0
                clearance = 5.0
                legal = True
                for _ in range(max(1, int(self.horizon/self.dt))):
                    yaw += turn*self.dt
                    x += velocity*math.cos(yaw)*self.dt
                    y += velocity*math.sin(yaw)*self.dt
                    if points.size:
                        clearance = min(clearance, float(np.min(np.linalg.norm(points-[x, y], axis=1)))-self.robot_radius)
                        if clearance < p['clearance_margin']:
                            legal = False
                            break
                if not legal:
                    continue
                heading_error = abs(self._wrap(theta-yaw))
                progress = velocity*math.cos(heading_error)
                score = (p['progress_weight']*progress + p['clearance_weight']*min(clearance,2.)
                         - p['heading_weight']*heading_error - p['turn_weight']*abs(turn))
                if score > best_score:
                    best_score, best = score, np.asarray([velocity, turn], dtype=np.float32)
        return best
