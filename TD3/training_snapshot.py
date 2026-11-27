"""Durable, generation-level training snapshots, with legacy read compatibility.

Only the atomically replaced index commits a generation. The files in the run
root are compatibility exports; recovery always resolves one immutable directory.
This module deliberately has no torch/ROS dependency.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import warnings


STORE = '.training_snapshots'
INDEX = 'snapshot_index.json'
BACKUP_INDEX = 'snapshot_index.previous.json'
VERSION = 1
CORE = ('checkpoint.pt', 'train_metrics.json')
_verified = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path):
    # Directory fsync is supported by the production Linux filesystem. Windows
    # has no equivalent through os.open; file contents still use fsync there.
    if os.name != 'nt':
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2,
                                 allow_nan=False).encode('utf-8'))


def has_checkpoint(run):
    run = Path(run)
    return any((run / name).exists() for name in (INDEX, BACKUP_INDEX, 'checkpoint.pt'))


@contextmanager
def training_lock(run):
    """One writer per run, released by the OS even after a power interruption."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.training.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                if handle.tell() == 0:
                    handle.write(b'0'); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another trainer owns this run: ' + str(run)) from error
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _generation(run, name, require_replay, verify):
    if not isinstance(name, str) or not re.fullmatch(r'step_\d+_[0-9a-f]{32}', name):
        raise ValueError('Invalid snapshot generation identifier')
    directory = run / STORE / name
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != VERSION or manifest['generation'] != name:
        raise ValueError('Snapshot manifest version/generation mismatch')
    files = manifest['files']
    for filename in CORE + (('replay.pkl',) if require_replay else ()):
        if filename not in files:
            raise ValueError('Snapshot lacks required ' + filename)
    for filename, record in files.items():
        if filename not in CORE + ('replay.pkl', 'rng.pkl'):
            raise ValueError('Unexpected snapshot filename')
        path = directory / filename
        if not path.is_file() or path.stat().st_size != record['size']:
            raise ValueError('Incomplete snapshot file: ' + str(path))
        if verify and sha256(path) != record['sha256']:
            raise ValueError('Snapshot checksum mismatch: ' + str(path))
    metrics = json.loads((directory / 'train_metrics.json').read_text(encoding='utf-8'))
    if metrics['metadata'] != manifest['metadata']:
        raise ValueError('Snapshot metrics/model metadata generation mismatch')
    if manifest['metadata'].get('snapshot_generation') != name:
        raise ValueError('Missing snapshot generation in metadata')
    return dict(directory=directory, metadata=manifest['metadata'], manifest=manifest,
                generation=name, legacy=False)


def resolve_snapshot(run, require_replay=False, verify=True):
    """Resolve a complete committed generation, falling back only within indexes.

    Orphan/pending directories never become recovery candidates. A managed run
    with no valid generation fails closed instead of mixing its root exports.
    Legacy files remain supported and are structurally checked by the trainer.
    """
    run = Path(run)
    managed = False
    failures, seen = [], set()
    for filename in (INDEX, BACKUP_INDEX):
        path = run / filename
        if not path.exists():
            continue
        managed = True
        try:
            index = json.loads(path.read_text(encoding='utf-8'))
            if index['version'] != VERSION:
                raise ValueError('Unsupported snapshot index version')
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(error)); continue
        for name in (index.get('current'), index.get('previous')):
            if name is None or name in seen:
                continue
            seen.add(name)
            try:
                result = _generation(run, name, require_replay, verify)
                if failures:
                    warnings.warn('Recovered previous complete snapshot at step %s: %s' %
                                  (result['metadata']['total_steps'], '; '.join(failures)))
                result['recovery_warnings'] = failures
                return result
            except (ValueError, KeyError, OSError, TypeError) as error:
                failures.append(str(error))
    if managed:
        raise RuntimeError('No valid committed training snapshot: ' + '; '.join(failures))
    for filename in CORE + (('replay.pkl',) if require_replay else ()):
        if not (run / filename).is_file():
            raise FileNotFoundError('Missing legacy training file: ' + str(run / filename))
    metrics = json.loads((run / 'train_metrics.json').read_text(encoding='utf-8'))
    return dict(directory=run, metadata=metrics['metadata'], manifest=None,
                generation=None, legacy=True, recovery_warnings=[])


def publish_exports(run, snapshot):
    """Repair evaluator/monitor paths without modifying immutable snapshot files."""
    run = Path(run)
    if snapshot['legacy']:
        return
    for filename in snapshot['manifest']['files']:
        if filename == 'rng.pkl':
            continue
        source = snapshot['directory'] / filename
        target = run / filename
        if target.exists() and os.path.samefile(source, target):
            continue
        temporary = run / ('.' + filename + '.' + uuid.uuid4().hex + '.export')
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
                with temporary.open('r+b') as handle:
                    os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    sync_directory(run)


def commit_snapshot(run, metadata, writers):
    """Write every component, flush it, then commit the entire generation once."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    sync_directory(run.parent)
    store = run / STORE
    store.mkdir(exist_ok=True)
    sync_directory(run)
    if not all(name in writers for name in CORE):
        raise ValueError('A snapshot requires both policy and metrics writers')
    previous = None
    if (run / INDEX).exists() or (run / BACKUP_INDEX).exists():
        previous = resolve_snapshot(run, require_replay='replay.pkl' in writers)['generation']
    name = 'step_%012d_%s' % (int(metadata['total_steps']), uuid.uuid4().hex)
    metadata = dict(metadata, snapshot_generation=name, snapshot_format_version=VERSION)
    staging = store / ('.pending_' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        records = {}
        for filename, writer in writers.items():
            if filename not in CORE + ('replay.pkl', 'rng.pkl'):
                raise ValueError('Unexpected snapshot filename')
            path = staging / filename
            writer(str(path), metadata)
            with path.open('r+b') as handle:
                os.fsync(handle.fileno())
            records[filename] = dict(size=path.stat().st_size, sha256=sha256(path))
        atomic_json(staging / 'manifest.json', dict(version=VERSION, generation=name,
                    metadata=metadata, files=records))
        sync_directory(staging)
        os.replace(staging, store / name)
        sync_directory(store)
        # Keep a small independent index for recovery from damaged index contents.
        if previous:
            atomic_json(run / BACKUP_INDEX, dict(version=VERSION, current=previous, previous=None))
        atomic_json(run / INDEX, dict(version=VERSION, current=name, previous=previous))
        result = _generation(run, name, 'replay.pkl' in writers, verify=False)
        publish_exports(run, result)
        # Only after the new pointer and exports are durable may older snapshots
        # be removed. Interrupted saves never delete the last committed state.
        for child in store.iterdir():
            if child.name not in (name, previous) and (child.name.startswith('.pending_') or
                    re.fullmatch(r'step_\d+_[0-9a-f]{32}', child.name)):
                if child.is_symlink() or child.resolve().parent != store.resolve():
                    raise ValueError('Refusing snapshot cleanup outside its storage directory')
                shutil.rmtree(child)
        sync_directory(store)
        print('[Snapshot] committed step %d (%s); previous=%s' %
              (metadata['total_steps'], name, previous), flush=True)
        return result
    except BaseException:
        # Leave unfinished writes for diagnosis/cleanup on the next successful
        # commit. They are never considered for recovery.
        raise


def saved_steps(run):
    """Fast progress query; full data verification is deferred until completion."""
    run = Path(run)
    if not has_checkpoint(run) and not (run / 'train_metrics.json').exists():
        return 0
    try:
        return int(resolve_snapshot(run, verify=False)['metadata']['total_steps'])
    except FileNotFoundError:
        # Legacy runs publish metrics before replay on their first save.
        path = run / 'train_metrics.json'
        return int(json.loads(path.read_text(encoding='utf-8'))['metadata']['total_steps']) if path.exists() else 0
    except json.JSONDecodeError:
        # A still-running legacy writer may be replacing its metrics file.
        if not (run / INDEX).exists() and not (run / BACKUP_INDEX).exists():
            return 0
        raise


def completed_run(run, budget):
    run = Path(run)
    if saved_steps(run) < budget:
        return False
    if not any((run / name).exists() for name in (INDEX, BACKUP_INDEX)):
        return all((run / name).is_file() for name in CORE + ('replay.pkl',))
    index_files = [run / INDEX, run / BACKUP_INDEX]
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in index_files if p.exists())
    # Do not rehash all completed replay pools every queue polling cycle.
    cached = _verified.get(str(run.resolve()))
    if cached and cached[0] == signature:
        candidate = cached[1]
        file_signature = tuple((p.stat().st_size, p.stat().st_mtime_ns)
                               for p in candidate['directory'].iterdir() if p.is_file())
        if file_signature == cached[2]:
            return candidate['metadata']['total_steps'] >= budget
    candidate = resolve_snapshot(run, require_replay=True)
    publish_exports(run, candidate)
    file_signature = tuple((p.stat().st_size, p.stat().st_mtime_ns)
                           for p in candidate['directory'].iterdir() if p.is_file())
    _verified[str(run.resolve())] = (signature, candidate, file_signature)
    return candidate['metadata']['total_steps'] >= budget
