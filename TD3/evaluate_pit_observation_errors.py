"""Evaluate frozen HRG-TD3 policies with point-cloud perception errors."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import threading
import numpy as np
import torch
import sensor_msgs.point_cloud2 as pc2
from evaluate_pass_threshold import threshold_environment, validate_checkpoint_metadata
from experiment_config import get_preset
from metrics_collector import MetricsCollector
from navigation_reward import PROTOCOL_VERSION, with_timeout_penalty
from policy_factory import make_learning_policy, policy_to_env_action
from pit_perception import signed_sector_observation
from pit_observation_errors import PointCloudError, CONDITIONS, geometry, floor_reference
from velodyne_env import GazeboRGBEnv
from arena_config import world_sha256

def atomic(path, data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            h.update(chunk)
    return h.hexdigest()

BaseEnvironment=threshold_environment(GazeboRGBEnv,.035)

class ErrorEnvironment(BaseEnvironment):
    def __init__(self, condition, *args, **kwargs):
        self.error_model=PointCloudError(condition)
        self.control_tick=-1
        self.episode_index=-1
        self.truth=None
        self.cloud_lock=threading.RLock()
        self.last_error={}
        self.gate_passable_height_m=.035
        self.callback_error=None
        super().__init__(*args,**kwargs)
        if self.callback_error:
            raise RuntimeError(self.callback_error)

    def raw_observation(self, points):
        heights,_,_=geometry(points,(self.sensor_x,self.sensor_y,self.robot_yaw),self.odom_vertical_row)
        floor=floor_reference(points,heights)
        sectors=[[] for _ in range(self.environment_dim)]
        angles=np.arctan2(points[:,1],points[:,0])
        ranges=np.linalg.norm(points[:,:2],axis=1)
        for i,(lo,hi) in enumerate(self.gaps):
            indices=(angles>=lo)&(angles<hi)&(ranges>1e-6)
            sectors[i]=np.column_stack((ranges[indices],heights[indices])).tolist()
        return signed_sector_observation(sectors,floor,ground_tol=.01,pit_depth_tol=.11,
            project_to_ground=True,sensor_height=.25,associate_nearest=True)

    def velodyne_callback(self, cloud):
        try:
            self._consume_cloud(cloud)
        except Exception as error:
            self.callback_error=repr(error)
            raise

    def _consume_cloud(self, cloud):
        with self.cloud_lock:
            if self.error_model.config['kind']=='clean':
                BaseEnvironment.velodyne_callback(self,cloud)
                self.truth=copy.copy(self)
                self.truth.pit_memory=copy.deepcopy(self.pit_memory)
                self.truth.velodyne_data=self.velodyne_data.copy()
                self.truth.obstacle_heights=self.obstacle_heights.copy()
                front=BaseEnvironment._front_obstacle(self)[2]
                self.last_error=dict(active=False,event_start=None,tick=self.control_tick,
                    removed=0,added=0,candidates=0,raw_missed_bins=0,raw_false_bins=0,
                    clean_front=front,error_front=front,front_missed=False,false_pit=False)
                return
            if self.truth is None:
                self.truth=copy.copy(self)
                self.truth.pit_memory=copy.deepcopy(self.pit_memory)
                self.truth.pit_front_latch=None
            for name in ('sensor_x','sensor_y','robot_yaw','odom_vertical_row'):
                setattr(self.truth,name,getattr(self,name))
            BaseEnvironment.velodyne_callback(self.truth,cloud)
            points=np.asarray(list(pc2.read_points(cloud,skip_nans=True,field_names=('x','y','z'))),dtype=np.float64).reshape(-1,3)
            changed,info=self.error_model.apply(points,(self.sensor_x,self.sensor_y,self.robot_yaw),
                self.odom_vertical_row,self.control_tick)
            if info['active']:
                altered=pc2.create_cloud_xyz32(cloud.header,changed)
            else:
                altered=cloud
            BaseEnvironment.velodyne_callback(self,altered)
            clean_front=BaseEnvironment._front_obstacle(self.truth)[2]
            error_front=BaseEnvironment._front_obstacle(self)[2]
            info.update(tick=self.control_tick,clean_front=clean_front,error_front=error_front,
                front_missed=clean_front=='pit' and error_front!='pit',
                false_pit=clean_front!='pit' and error_front=='pit')
            if info['active']:
                before_d,before_h=self.raw_observation(points)
                after_d,after_h=self.raw_observation(changed)
                before=(before_h<-.11)&(before_d<2)
                after=(after_h<-.11)&(after_d<2)
                info['raw_missed_bins']=int(np.count_nonzero(before & ~after))
                info['raw_false_bins']=int(np.count_nonzero(~before & after))
            else:
                info.update(raw_missed_bins=0,raw_false_bins=0)
            self.last_error=info

    def observe_collision(self, laser_data):
        # The original collision rule reads the uncorrupted shadow observation.
        # The policy, gate, shield and pit memory consume the corrupted cloud.
        with self.cloud_lock:
            if self.truth is None:
                return super().observe_collision(laser_data)
            return BaseEnvironment.observe_collision(self.truth,self.truth.velodyne_data)

    def reset(self, scenario=None):
        with self.cloud_lock:
            self.control_tick=-1
            self.truth=None
            self.error_model.reset(-1)
        state,image=super().reset(scenario=scenario)
        if self.callback_error:
            raise RuntimeError(self.callback_error)
        with self.cloud_lock:
            self.truth=copy.copy(self)
            self.truth.pit_memory=copy.deepcopy(self.pit_memory)
            self.truth.velodyne_data=self.velodyne_data.copy()
            self.truth.obstacle_heights=self.obstacle_heights.copy()
        self.episode_index+=1
        self.error_model.reset(self.episode_index)
        self.control_tick=0
        self.last_error={}
        return state,image

    def step(self,action):
        self.control_tick+=1
        state,image,reward,done,info=super().step(action)
        if self.callback_error:
            raise RuntimeError(self.callback_error)
        with self.cloud_lock:
            error=dict(self.last_error)
        info.update(perception_error=error)
        return state,image,reward,done,info

def make_environment(condition):
    cfg=get_preset('full')
    return ErrorEnvironment(condition,'paper_traversability.launch',cfg.n_sectors,
        robot_name='p3dx',connect_only=True,use_height=cfg.use_height,use_rgb=False,
        use_pass_gate=cfg.use_pass_gate,use_safety_shield=cfg.use_safety_shield,
        use_enhanced_features=False,robot_dim=cfg.robot_dim,reward_mode=cfg.reward_mode,
        passable_height=cfg.physical_passable_height,safety_stop_distance=cfg.safety_stop_distance,
        safety_slow_distance=cfg.safety_slow_distance,pass_turn_damping=cfg.pass_turn_damping,
        pass_usage_cost=cfg.pass_usage_cost)

def preflight(plan):
    env=make_environment('clean')
    results={}
    poses=dict(miss=[-6.,1.95,0.],false=[0.,0.,0.])
    for kind,pose in poses.items():
        env.reset(scenario=dict(robot_pose=pose,goal=[pose[0]+3.,pose[1]]))
        # Capture one fresh scan while holding a zero velocity command.
        env.step([0.,0.,-1.])
        import rospy
        from sensor_msgs.msg import PointCloud2
        env.unpause()
        cloud=rospy.wait_for_message('/velodyne_points',PointCloud2,timeout=10)
        env.pause()
        points=np.asarray(list(pc2.read_points(cloud,skip_nans=True,field_names=('x','y','z'))),dtype=np.float64)
        current_pose=(env.sensor_x,env.sensor_y,env.robot_yaw)
        for condition in (['miss30','miss60'] if kind=='miss' else ['false3','false10']):
            model=PointCloudError(condition)
            model.reset(0)
            before_d,before_h=env.raw_observation(points)
            rows=[]
            for tick in range(1,23):
                changed,info=model.apply(points,current_pose,env.odom_vertical_row,tick)
                after_d,after_h=env.raw_observation(changed)
                before=(before_h<-.11)&(before_d<2)
                after=(after_h<-.11)&(after_d<2)
                info.update(raw_missed_bins=int(np.count_nonzero(before&~after)),
                    raw_false_bins=int(np.count_nonzero(~before&after)))
                rows.append(info)
            active=[r for r in rows if r['active']]
            assert len(active)==CONDITIONS[condition]['duration'],(condition,rows)
            key='raw_missed_bins' if kind=='miss' else 'raw_false_bins'
            assert max(r[key] for r in active)>0,(condition,rows)
            results[condition]=dict(passed=True,pose=list(current_pose),active_steps=len(active),
                maximum_changed_pit_sectors=max(r[key] for r in active),sample=active[0])
            clean_collision=env.observe_collision(env.velodyne_data)[1]
            env.error_model=PointCloudError(condition)
            env.error_model.reset(0)
            env.control_tick=1 if kind=='miss' else 10
            env._consume_cloud(cloud)
            pipeline=dict(env.last_error)
            if kind=='false':
                assert pipeline['false_pit'],pipeline
            assert env.observe_collision(env.velodyne_data)[1]==clean_collision
            assert env.callback_error is None
            results[condition]['full_pipeline']=pipeline
            results[condition]['collision_rule_preserved']=True
            env.error_model=PointCloudError('clean')
            env.pit_memory.clear()
            env.pit_front_latch=None
            env._consume_cloud(cloud)
    atomic(Path(plan['run_root'])/'preflight.json',dict(passed=True,conditions=results))
    print(json.dumps(results),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--condition',choices=CONDITIONS)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    torch.set_num_threads(4)
    if args.preflight:
        preflight(plan)
        return
    source=next(s for s in plan['sources'] if s['seed']==args.seed)
    assert digest(source['checkpoint'])==source['checkpoint_sha256']
    assert digest(plan['evaluation']['scenario_file'])==plan['evaluation']['scenario_file_sha256']
    assert world_sha256('paper_traversability.launch')==plan['evaluation']['world_sha256']
    run=Path(plan['run_root'])/('%s_seed%d'%(args.condition,args.seed))
    run.mkdir(parents=True,exist_ok=True)
    assert not (run/'result.json').exists()
    cfg=get_preset('full')
    random.seed(100000)
    np.random.seed(100000)
    torch.manual_seed(100000)
    env=make_environment(args.condition)
    policy=make_learning_policy(cfg,log_dir=str(run))
    metadata=policy.load_training_state(str(source['checkpoint']),load_optimizers=False)
    validate_checkpoint_metadata(metadata,cfg.to_dict(),args.seed,PROTOCOL_VERSION)
    if metadata!=source['metadata']:
        raise ValueError('Source checkpoint metadata changed')
    scenarios=json.loads(Path(plan['evaluation']['scenario_file']).read_text())['scenarios'][:100]
    collector=MetricsCollector()
    trace=(run/'trace.jsonl').open('w',encoding='utf-8')
    journal=(run/'episodes.jsonl').open('w',encoding='utf-8')
    try:
        for episode,scenario in enumerate(scenarios):
            state,image=env.reset(scenario=scenario)
            collector.reset_episode(env.prev_distance)
            counts=dict(event_count=0,injected_steps=0,removed_points=0,added_points=0,
                candidate_points_during_injection=0,raw_missed_sectors=0,raw_false_sectors=0,
                final_front_miss_steps=0,final_false_pit_steps=0)
            started=False
            for tick in range(500):
                raw=policy.get_action(state,image)
                weights=policy.get_recal_weights()
                action=policy_to_env_action(raw,cfg,state)
                state,image,reward,done,info=env.step(action)
                reward=with_timeout_penalty(reward,not done and tick==499,cfg.reward_mode)
                info['recal_weights']=weights
                collector.record_step(reward,done,info)
                error=info['perception_error']
                if error.get('event_start') is not None and not started:
                    started=True
                    counts['event_count']=1
                counts['injected_steps']+=int(error.get('active',False))
                for target,key in [('removed_points','removed'),('added_points','added'),
                    ('raw_missed_sectors','raw_missed_bins'),('raw_false_sectors','raw_false_bins')]:
                    counts[target]+=int(error.get(key,0))
                if error.get('active'):
                    counts['candidate_points_during_injection']+=error.get('candidates',0)
                counts['final_front_miss_steps']+=int(error.get('front_missed',False))
                counts['final_false_pit_steps']+=int(error.get('false_pit',False))
                row={k:v for k,v in info.items() if k!='recal_weights'}
                trace.write(json.dumps(dict(episode=episode,step=tick,**row))+'\n')
                if done:
                    break
            result=collector.finish_episode(timeout=not done)
            result.update(counts)
            result['episode']=episode
            journal.write(json.dumps(result)+'\n')
            journal.flush()
            trace.flush()
            atomic(run/'progress.json',dict(condition=args.condition,seed=args.seed,
                completed=episode+1,expected=100,statistics=collector.statistics(),last_episode=result))
            print(json.dumps(dict(episode=episode+1,statistics=collector.statistics(),errors=counts)),flush=True)
        meta=dict(condition=args.condition,settings=CONDITIONS[args.condition],gate_threshold_m=.035,
            checkpoint_sha256=source['checkpoint_sha256'],checkpoint_metadata=metadata,
            scenario_file_sha256=plan['evaluation']['scenario_file_sha256'],
            world_sha256=world_sha256('paper_traversability.launch'),plan_sha256=digest(args.plan),
            model_seed=args.seed,evaluation_seed=100000,max_episode_steps=500,
            collision_observer='uncorrupted cloud through original collision rule',
            pit_fall_criterion='physical robot_z < -0.08',memory='original pit memory and latch active')
        if digest(source['checkpoint'])!=source['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
        atomic(run/'result.json',dict(metadata=meta,statistics=collector.statistics(),episodes=collector.episodes))
    finally:
        trace.close()
        journal.close()
        policy.writer.close()

if __name__=='__main__':
    main()
