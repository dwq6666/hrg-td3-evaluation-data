"""Point-cloud corruption used by the fixed-policy pit robustness evaluation."""
import math
import numpy as np
from arena_config import V6_PITS, valid_v6_position

CONDITIONS = {
    'clean': dict(kind='clean', fraction=0., duration=0),
    'miss30': dict(kind='miss', fraction=.30, duration=10),
    'miss60': dict(kind='miss', fraction=.60, duration=10),
    'false3': dict(kind='false', fraction=0., duration=3),
    'false10': dict(kind='false', fraction=0., duration=10),
}

def geometry(points, pose, vertical):
    x, y, yaw = pose
    height = ((points[:, 0]+.125)*vertical[0] + points[:, 1]*vertical[1]
              + (points[:, 2]+.25)*vertical[2])
    wx = x+(points[:, 0]+.125)*math.cos(yaw)-points[:, 1]*math.sin(yaw)
    wy = y+(points[:, 0]+.125)*math.sin(yaw)+points[:, 1]*math.cos(yaw)
    return height, wx, wy

def floor_reference(points, heights):
    ranges = np.linalg.norm(points[:, :2], axis=1)
    selected = (ranges>.35)&(ranges<3)&(heights>=-.18)&(heights<=.18)
    return float(np.median(heights[selected])) if selected.any() else 0.

def pit_candidates(points, pose, vertical):
    heights, wx, wy = geometry(points, pose, vertical)
    floor = floor_reference(points, heights)
    footprint = np.zeros(len(points), dtype=bool)
    for xmin, xmax, ymin, ymax in V6_PITS:
        footprint |= (wx>=xmin-.05)&(wx<=xmax+.05)&(wy>=ymin-.05)&(wy<=ymax+.05)
    angle = np.arctan2(points[:, 1], points[:, 0])
    selected = footprint & (heights<floor-.11) & (np.abs(angle)<math.pi/2)
    ranges = np.linalg.norm(points[:, :2], axis=1)
    edge = np.maximum(0, ranges*.25/(.25+np.maximum(floor-heights, 1e-5))-.25)
    eligible = bool(np.any(selected & (np.abs(angle)<.4) & (edge<1.5)))
    return selected, floor, eligible

def flat_patch(points, pose, vertical):
    heights, _, _ = geometry(points, pose, vertical)
    floor = floor_reference(points, heights)
    patch = (points[:, 0]>=1.1)&(points[:, 0]<=1.5)&(np.abs(points[:, 1])<.22)
    if np.count_nonzero(patch & (np.abs(heights-floor)<.04))<8:
        return None
    if np.any(patch & (np.abs(heights-floor)>.08)):
        return None
    x, y, yaw = pose
    for px in [1.1, 1.3, 1.5]:
        for py in [-.22, 0., .22]:
            wx=x+(px+.125)*math.cos(yaw)-py*math.sin(yaw)
            wy=y+(px+.125)*math.sin(yaw)+py*math.cos(yaw)
            if not valid_v6_position(wx, wy):
                return None
    return floor

class PointCloudError:
    def __init__(self, condition):
        self.config=CONDITIONS[condition]
        self.reset(-1)

    def reset(self, episode):
        self.episode=episode
        self.start=None
        self.false_world=None
        self.anchor=None

    def apply(self, points, pose, vertical, tick):
        selected, floor, eligible=pit_candidates(points, pose, vertical)
        info=dict(active=False, candidates=int(selected.sum()), removed=0, added=0,
                  event_start=self.start, event_started=False, intended_fraction=self.config['fraction'])
        kind=self.config['kind']
        if kind=='clean' or tick<1 or self.episode<0:
            return points, info
        if self.start is None:
            if kind=='miss' and eligible:
                self.start=tick
                # Seeded angular anchor gives correlated spatial occlusion.
                # Severity changes the retained point count using the same anchor.
                angles=np.arctan2(points[selected, 1], points[selected, 0])
                rng=np.random.RandomState(831700+self.episode)
                self.anchor=float(np.quantile(angles, rng.uniform(.35, .65)))
            elif kind=='false' and tick>=10:
                floor=flat_patch(points, pose, vertical)
                if floor is not None:
                    self.start=tick
                    px, py=np.meshgrid(np.linspace(1.1,1.5,8), np.linspace(-.22,.22,8))
                    x, y, yaw=pose
                    self.false_world=np.column_stack((
                        x+(px.ravel()+.125)*math.cos(yaw)-py.ravel()*math.sin(yaw),
                        y+(px.ravel()+.125)*math.sin(yaw)+py.ravel()*math.cos(yaw)))
            if self.start is not None:
                info['event_started']=True
        info['event_start']=self.start
        if self.start is None or not self.start<=tick<self.start+self.config['duration']:
            return points, info
        info['active']=True
        if kind=='miss':
            indices=np.flatnonzero(selected)
            angles=np.arctan2(points[indices,1], points[indices,0])
            count=int(round(len(indices)*self.config['fraction']))
            chosen=indices[np.argsort(np.abs(angles-self.anchor),kind='stable')[:count]]
            keep=np.ones(len(points),dtype=bool)
            keep[chosen]=False
            info['removed']=len(chosen)
            return points[keep], info
        x, y, yaw=pose
        dx=self.false_world[:,0]-x
        dy=self.false_world[:,1]-y
        px=dx*math.cos(yaw)+dy*math.sin(yaw)-.125
        py=-dx*math.sin(yaw)+dy*math.cos(yaw)
        pz=(floor-.30-vertical[0]*(px+.125)-vertical[1]*py)/vertical[2]-.25
        fake=np.column_stack((px,py,pz))
        info['added']=len(fake)
        return np.concatenate((points,fake)),info
