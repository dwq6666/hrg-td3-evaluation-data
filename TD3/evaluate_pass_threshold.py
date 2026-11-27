"""Evaluate frozen full policies with an explicit, isolated pass-authorization override.

All ordinary evaluation arguments are delegated to test_velodyne_td3.py.  This
adapter deliberately does not edit a preset, checkpoint, world or collision
criterion: only the low/high class used by the learned pass authorization is
overridden.  The controller must launch an isolated simulator before this worker.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest import mock


EXPERIMENT_VERSION = 'pass_authorization_threshold_v1'
THRESHOLDS = (.020, .022, .035, .050)
FIXED_COLLISION_HEIGHT_M = .025
FIXED_RECALIBRATION_HEIGHT_M = .025
BOUNDARY = 'observed_height < threshold'


def threshold_value(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Gate threshold must be finite')
    for allowed in THRESHOLDS:
        if math.isclose(value, allowed, rel_tol=0., abs_tol=1e-12):
            return allowed
    raise ValueError('Approved gate thresholds are 0.020, 0.022, 0.035 and 0.050 m')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def checked_digest(path, expected):
    value = digest(path)
    if value != expected:
        raise ValueError('Source file checksum mismatch: ' + str(path))
    return value


def override_front_class(result, threshold):
    """Retain base selection/pit/none semantics, changing only ordinary low/high."""
    distance, height, category = result
    if category in ('pit', 'none'):
        return result
    if category not in ('low', 'high'):
        raise ValueError('Unexpected base obstacle class: ' + str(category))
    return distance, height, ('low' if height < threshold else 'high')


def threshold_environment(base_class, threshold):
    """Create a narrow subclass; collision and every other base method stay intact."""
    threshold = threshold_value(threshold)

    class GateThresholdEnvironment(base_class):
        def __init__(self, *args, **kwargs):
            collision_height = kwargs.get('passable_height', FIXED_COLLISION_HEIGHT_M)
            if not math.isclose(collision_height, FIXED_COLLISION_HEIGHT_M, rel_tol=0., abs_tol=1e-12):
                raise ValueError('Collision evaluation height must remain 0.025 m')
            super().__init__(*args, **kwargs)
            self.gate_passable_height_m = threshold

        def _front_obstacle(self, distance=None, height=None):
            base = super()._front_obstacle(distance, height)
            return override_front_class(base, self.gate_passable_height_m)

    return GateThresholdEnvironment


def validate_checkpoint_metadata(metadata, config, seed, protocol):
    if metadata.get('preset') != 'full' or metadata.get('seed') != seed:
        raise ValueError('Checkpoint must be the planned full policy and training seed')
    if metadata.get('total_steps') != 500000:
        raise ValueError('Checkpoint must be the final 500000-step source')
    if metadata.get('config') != config:
        raise ValueError('Checkpoint configuration differs from the unchanged full preset')
    if metadata.get('protocol_version') != protocol:
        raise ValueError('Checkpoint training protocol mismatch')
    arguments = metadata.get('arguments', {})
    if (arguments.get('launchfile') != 'paper_traversability.launch'
            or arguments.get('robot_name') != 'p3dx'
            or arguments.get('max_timesteps') != 500000
            or arguments.get('max_episode_steps') != 500):
        raise ValueError('Checkpoint training origin/budget differs from this experiment')


def runtime_metadata(metadata, threshold, checkpoint, checkpoint_sha, scenario_sha,
                     world_sha, adapter_sha, trace_path=None):
    result = dict(metadata)
    result.update(
        threshold_experiment_version=EXPERIMENT_VERSION,
        gate_passable_height_m=threshold_value(threshold),
        gate_boundary=BOUNDARY,
        collision_passable_height_m=FIXED_COLLISION_HEIGHT_M,
        recalibration_height_m=FIXED_RECALIBRATION_HEIGHT_M,
        checkpoint_sha256=checkpoint_sha,
        source_checkpoint=str(Path(checkpoint).resolve()),
        source_scenario_file_sha256=scenario_sha,
        source_world_sha256=world_sha,
        evaluator_adapter_sha256=adapter_sha,
        runtime_override=dict(parameter='gate_passable_height_m', value_m=threshold_value(threshold),
                              trained_configuration_unchanged=True),
    )
    if trace_path is not None:
        result['trace_file'] = str(Path(trace_path).resolve())
        result['trace_sha256'] = digest(trace_path)
        result['trace_bytes'] = Path(trace_path).stat().st_size
    return result


def _load_metadata(torch_module, checkpoint):
    try:
        saved = torch_module.load(str(checkpoint), map_location='cpu', weights_only=False)
    except TypeError:
        saved = torch_module.load(str(checkpoint), map_location='cpu')
    return saved['metadata']


def override_argument_parser():
    # --checkpoint belongs to the ordinary evaluator.  Prefix abbreviation
    # would otherwise consume it as --checkpoint-source-sha256 here.
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False, description=__doc__)
    extra.add_argument('--gate-passable-height', type=threshold_value, required=True)
    extra.add_argument('--checkpoint-source-sha256', required=True)
    return extra


def parse_override_arguments(argv):
    return override_argument_parser().parse_known_args(argv)


def main():
    extra = override_argument_parser()
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        extra.print_help()
        print('All other arguments use the unchanged test_velodyne_td3.py interface. '
              'Specify --preset full, --seed, --checkpoint, --scenario-file, '
              '--run-root, --run-name and --trace-episodes 100; omit --launch-sim.')
        return
    override, standard_arguments = parse_override_arguments(sys.argv[1:])

    # Importing the ordinary worker requires ROS.  Keep imports lazy so the
    # geometry/metadata adapter can be tested without ROS or a simulator.
    import test_velodyne_td3 as standard
    with mock.patch.object(sys, 'argv', [sys.argv[0]] + standard_arguments):
        args = standard.parse_args()
    if args.preset != 'full' or args.profile != 'nominal':
        raise ValueError('This experiment supports unchanged full policies under nominal conditions only')
    if args.seed not in (0, 1, 2):
        raise ValueError('This experiment uses training seeds 0, 1 and 2 only')
    if args.launchfile != 'paper_traversability.launch' or args.robot_name != 'p3dx':
        raise ValueError('This experiment uses the original paper_traversability P3-DX scene only')
    if args.launch_sim:
        raise ValueError('The queue owns the isolated simulator; omit --launch-sim')
    if not args.checkpoint or not args.scenario_file or not args.run_name:
        raise ValueError('Explicit source checkpoint, frozen scenario file and unique run-name are required')
    if args.episodes != 100 or args.max_episode_steps != 500 or args.eval_seed != 100000:
        raise ValueError('Formal experiment requires 100 episodes, a 500-step limit and eval seed 100000')
    if args.trace_episodes != args.episodes:
        raise ValueError('All 100 episodes must be traced')
    cfg = standard.get_preset('full')
    if not math.isclose(cfg.physical_passable_height, FIXED_COLLISION_HEIGHT_M, abs_tol=1e-12):
        raise ValueError('Frozen collision configuration changed')
    if not math.isclose(cfg.passable_height, FIXED_RECALIBRATION_HEIGHT_M, abs_tol=1e-12):
        raise ValueError('Frozen recalibration reference changed')
    checkpoint = Path(args.checkpoint).resolve()
    scenario = Path(args.scenario_file).resolve()
    if not checkpoint.is_file() or not scenario.is_file():
        raise FileNotFoundError('The planned source checkpoint or frozen scenario file is missing')
    checkpoint_sha = checked_digest(checkpoint, override.checkpoint_source_sha256)
    source_metadata = _load_metadata(standard.torch, checkpoint)
    validate_checkpoint_metadata(source_metadata, cfg.to_dict(), args.seed, standard.PROTOCOL_VERSION)
    source_scenario_sha = digest(scenario)
    source_world_sha = standard.world_sha256(args.launchfile)
    adapter_sha = digest(__file__)
    run_dir = Path(args.run_root).resolve() / args.run_name
    if run_dir.resolve() == checkpoint.parent:
        raise ValueError('Evaluation output must not be the training source directory')
    scene = Path(args.launchfile).stem
    filename = 'evaluation_%s_%s_seed%d.json' % (scene, args.profile, args.eval_seed)
    output = run_dir / filename
    trace = run_dir / ('trace_%s_%s_seed%d.csv' % (scene, args.profile, args.eval_seed))
    if output.exists() or trace.exists():
        raise FileExistsError('Existing evaluation/trace must be validated or archived by the queue before launching')
    BaseCollector = standard.MetricsCollector

    class ThresholdMetricsCollector(BaseCollector):
        def save(self, filepath, metadata=None):
            if Path(filepath).resolve() != output.resolve():
                raise ValueError('Unexpected evaluation output path')
            checked_digest(checkpoint, checkpoint_sha)
            checked_digest(scenario, source_scenario_sha)
            if standard.world_sha256(args.launchfile) != source_world_sha:
                raise ValueError('World changed during evaluation')
            if digest(__file__) != adapter_sha:
                raise ValueError('Threshold adapter changed during evaluation')
            if metadata['checkpoint_metadata'] != source_metadata or metadata['method_config'] != cfg.to_dict():
                raise ValueError('Evaluation did not preserve the source model identity/configuration')
            enriched = runtime_metadata(metadata, override.gate_passable_height, checkpoint,
                checkpoint_sha, source_scenario_sha, source_world_sha, adapter_sha, trace)
            temporary = output.with_name('.' + output.name + '.partial_' + str(os.getpid()))
            super().save(str(temporary), enriched)
            if output.exists():
                raise FileExistsError('Another worker created the evaluation output')
            os.replace(str(temporary), str(output))

    print('Threshold experiment: gate=%.4f m, collision=%.4f m, recalibration=%.4f m, strict <' %
          (override.gate_passable_height, FIXED_COLLISION_HEIGHT_M, FIXED_RECALIBRATION_HEIGHT_M), flush=True)
    Environment = threshold_environment(standard.GazeboRGBEnv, override.gate_passable_height)
    with mock.patch.object(standard, 'parse_args', return_value=args), \
            mock.patch.object(standard, 'GazeboRGBEnv', Environment), \
            mock.patch.object(standard, 'MetricsCollector', ThresholdMetricsCollector):
        standard.main()
    if not output.is_file():
        raise RuntimeError('Evaluation exited without a finalized result')
    print('Threshold result finalized: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
