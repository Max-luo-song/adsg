"""Waymo per-scene RGB prediction and AD scene graph training driver."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .common import records, signature, write_json

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('pose', 'depth', 'sam', 'semantic', 'traffic', 'assemble', 'validate', 'train')


def parse_scene_list(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        row = next(csv.reader([line]))
        try:
            scene = int(row[0].strip())
        except ValueError:
            raise ValueError(f'Expected Waymo numeric ID in {path}: {line!r}')
        if not 0 <= scene <= 797:
            raise ValueError(f'Waymo training scene ID out of range: {scene}')
        start = int(row[2]) if len(row) > 2 else 0
        end = int(row[3]) if len(row) > 3 else -1
        if start < 0 or (end != -1 and end < start):
            raise ValueError(f'Invalid frame range: {line}')
        result.append(dict(scene_idx=scene, start=start, end=end))
    if not result or len({x['scene_idx'] for x in result}) != len(result):
        raise ValueError('Scene list must be nonempty and unique')
    return result


def selected(cfg, only):
    scenes = parse_scene_list(cfg['scene_list'])
    if only:
        ids = set(int(x) for x in only.split(','))
        scenes = [s for s in scenes if s['scene_idx'] in ids]
        if {s['scene_idx'] for s in scenes} != ids:
            raise ValueError('--scenes contains IDs outside the configured reference split')
    return scenes


def index_scene(cfg, spec):
    source = Path(cfg['rgb_root']) / f"{spec['scene_idx']:03d}" / 'images'
    cameras = [str(c) for c in cfg['cameras']]
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= {'0','1','2','3','4'}:
        raise ValueError('Expected distinct Waymo camera IDs')
    if str(cfg.get('reference_camera', 0)) not in cameras or cfg['fps'] <= 0:
        raise ValueError('Reference camera must be selected and fps must be positive')
    maps = {c: {} for c in cameras}
    if not source.is_dir():
        raise FileNotFoundError(f'Missing RGB-only source directory: {source}')
    for image in source.iterdir():
        if image.suffix.lower() != '.jpg':
            continue
        fields = image.stem.split('_')
        if len(fields) == 2 and fields[0].isdigit() and fields[1] in maps:
            fid, cam = int(fields[0]), fields[1]
            if fid in maps[cam]:
                raise ValueError(f'Duplicate frame {fid}, camera {cam}')
            maps[cam][fid] = image.resolve()
    ids = sorted(maps[cameras[0]])
    if not ids or any(set(m) != set(ids) for m in maps.values()):
        raise ValueError(f'Unsynchronized/empty RGB cameras: {source}')
    end = ids[-1] if spec['end'] == -1 else spec['end']
    frame_ids = list(range(spec['start'], end + 1))
    if not frame_ids or not set(frame_ids) <= set(ids):
        raise ValueError(f'Missing RGB frames in requested inclusive range {spec["start"]}..{end}')
    frames = [dict(frame_id=fid, timestamp=fid / cfg['fps'],
                   images={c: str(maps[c][fid]) for c in cameras}) for fid in frame_ids]
    stats = [[str(p), p.stat().st_size, p.stat().st_mtime_ns]
             for fid in frame_ids for c in cameras for p in [maps[c][fid]]]
    scene = dict(schema_version=1, scene_name=f"{spec['scene_idx']:03d}",
                 scene_idx=spec['scene_idx'], camera_ids=cameras, reference_camera=str(cfg.get('reference_camera', 0)),
                 fps=cfg['fps'], frames=frames, sensor_inputs=['rgb'],
                 gt_intrinsics_used=False, gt_poses_used=False, gt_depth_used=False,
                 gt_boxes_used=False, gt_masks_used=False, rgb_stat_hash=signature(stats))
    scene['input_signature'] = signature(scene)
    folder = Path(cfg['output_root']) / scene['scene_name']
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'rgb_scene.json'
    if manifest.exists() and json.loads(manifest.read_text()) != scene:
        raise RuntimeError(f'Output belongs to different RGB inputs. Use a new output root: {folder}')
    write_json(manifest, scene)
    for fid, camera, image in records(scene):
        dest = folder / 'images' / f'{fid:03d}_{camera}.jpg'
        dest.parent.mkdir(exist_ok=True)
        if dest.is_symlink():
            if dest.resolve() != image:
                raise RuntimeError(f'Conflicting RGB link: {dest}')
        elif dest.exists():
            raise RuntimeError(f'Refusing to replace an existing image: {dest}')
        else:
            dest.symlink_to(image)
    write_json(folder / 'pose_manifest.json', dict(schema_version=1, scenes=[scene]))
    print(f"INDEX_OK scene={scene['scene_name']} frames={len(frames)} cameras={cameras}", flush=True)
    return folder


def stage_command(cfg, stage, folder):
    python = cfg.get('python_by_stage', {}).get(stage, cfg['python'])
    if stage == 'train':
        return [python, '-m', 'tools.train', '--config_file', str(ROOT / 'configs/omnire_rgb_only.yaml'),
                '--output_root', cfg['training_output'], '--project', f'waymo_rgb_{folder.name}',
                '--run_name', cfg['run_name'], f'data.data_root={folder.parent}',
                f'data.scene_idx={int(folder.name)}',
                'data.pixel_source.cameras=' + json.dumps(cfg['cameras'],separators=(',',':')),
                f'trainer.optim.num_iters={cfg["num_iters"]}']
    return [python, '-m', 'tools.rgb_only.stages', '--config', cfg['_path'],
            '--scene-dir', str(folder), '--stage', stage]


def run_stage(cfg, stage, folder):
    command = stage_command(cfg, stage, folder)
    print(f'[{folder.name}/{stage}] {shlex.join(command)}', flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='offline',
               CUDA_VISIBLE_DEVICES=str(cfg.get('gpu', '0')), PYTHONUNBUFFERED='1')
    if stage == 'train':
        from .stages import validate, claim
        token = claim(cfg,folder)
        marker = folder / '.stages/assemble.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('signature') != token:
            raise RuntimeError('A compatible completed assembly is required before training')
        validate(cfg, folder)
        run = Path(cfg['training_output']) / f'waymo_rgb_{folder.name}' / cfg['run_name']
        if run.exists():
            raise FileExistsError(f'Training run already exists; use a new run_name: {run}')
    logs = folder / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / f'{stage}.log').open('a', buffering=1) as log:
        log.write('\n$ ' + shlex.join(command) + '\n')
        log.flush()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = (logs / f'{stage}.log').read_text(errors='replace')[-9000:]
        raise RuntimeError(f'{stage} failed; log={logs / (stage + ".log")}\n{tail}')
    print(f'[{folder.name}/{stage}] OK log={logs / (stage + ".log")}', flush=True)


def doctor(cfg, specs):
    checks = []
    required = ('python', 'omega_checkpoint', 'moge_checkpoint', 'sam_checkpoint',
                'grounding_model', 'segformer_config', 'segformer_checkpoint',
                'traffic_config', 'traffic_checkpoint')
    for key in required:
        path = Path(cfg[key])
        checks.append(dict(check=key, ok=path.exists(), path=str(path)))
    for spec in specs:
        path = Path(cfg['rgb_root']) / f"{spec['scene_idx']:03d}" / 'images'
        checks.append(dict(check=f'rgb_scene_{spec["scene_idx"]:03d}', ok=path.is_dir(), path=str(path)))
    probes = {
        'pose': ['torch', 'torchvision', 'einops', 'cv2', 'utils3d', 'scipy'],
        'sam': ['torch', 'sam2', 'transformers'],
        'semantic': ['torch', 'mmseg', 'mmcv'],
        'traffic': ['torch', 'groundingdino', 'sklearn'],
        'train': ['torch', 'pytorch3d', 'gsplat', 'nvdiffrast', 'open3d', 'omegaconf', 'wandb']}
    for stage, modules in probes.items():
        python = cfg.get('python_by_stage', {}).get(stage, cfg['python'])
        paths = [str(ROOT)]
        if stage == 'semantic': paths.insert(0,cfg['segformer_repo'])
        if stage == 'sam': paths.insert(0,cfg['sam_repo'])
        if stage == 'traffic': paths.insert(0,str(ROOT / 'third_party/GroundingDINO'))
        code = ('import importlib,json,sys\n' + f'sys.path[:0]={paths!r}\n' +
                'result={}\n' + f'for name in {modules!r}:\n' +
                ' try:\n  importlib.import_module(name)\n  result[name]=True\n' +
                ' except Exception as exc:\n  result[name]=str(type(exc).__name__)+": "+str(exc)\n' +
                'print(json.dumps(result))')
        try:
            p = subprocess.run([python, '-c', code], capture_output=True, text=True, timeout=45)
            details=json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
            checks.append(dict(check=f'{stage}_imports', ok=details is not None and all(v is True for v in details.values()),
                               python=python, details=p.stdout.strip() or p.stderr[-1200:]))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(dict(check=f'{stage}_imports', ok=False, details=str(exc)))
    try:
        p = subprocess.run([cfg['python'], '-c', 'import torch; print(torch.cuda.is_available(),torch.cuda.device_count())'],
                           capture_output=True, text=True, timeout=30)
        checks.append(dict(check='cuda', ok=p.returncode == 0 and p.stdout.startswith('True'), details=p.stdout.strip() or p.stderr[-1000:]))
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.append(dict(check='cuda', ok=False, details=str(exc)))
    report = dict(ok=all(c['ok'] for c in checks), checks=checks)
    write_json(Path(cfg['output_root']) / 'doctor.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report['ok']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rgb_only/waymo.json')
    parser.add_argument('--stage', choices=('plan', 'index', 'doctor', 'all') + STAGES, default='plan')
    parser.add_argument('--scenes', help='Comma-separated numeric IDs within the reference split')
    a = parser.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg['_path'] = str(Path(a.config).resolve())
    specs = selected(cfg, a.scenes)
    if a.stage == 'plan':
        print(json.dumps(dict(scenes=specs, stages=STAGES, rgb_root=cfg['rgb_root'],
                              output_root=cfg['output_root'], rgb_only=True), indent=2))
        return
    if a.stage == 'doctor':
        sys.exit(0 if doctor(cfg, specs) else 2)
    if a.stage == 'all' and not doctor(cfg, specs):
        raise RuntimeError('Preflight failed. See doctor.json; no GPU jobs launched.')
    for spec in specs:
        folder = index_scene(cfg, spec)
        if a.stage == 'index':
            continue
        for stage in STAGES if a.stage == 'all' else (a.stage,):
            run_stage(cfg, stage, folder)


if __name__ == '__main__':
    main()
