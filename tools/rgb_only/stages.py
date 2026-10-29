"""Separate model processes with local assets and an explicit RGB-only contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile

import numpy as np

from .common import (WORLD_FROM_OPENCV, camera_table, key_name, read_scene, records,
                     resized_depth, resized_labels, save_npz, signature, unproject, write_json)

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = dict(pose=[], depth=['pose'], sam=[], semantic=[], traffic=['pose', 'depth'],
                    assemble=['pose', 'depth', 'sam', 'semantic', 'traffic'], validate=['assemble'])


def claim(cfg, folder):
    scene = read_scene(folder)
    assets = {}
    for key in ('omega_checkpoint', 'moge_checkpoint', 'sam_checkpoint', 'grounding_model',
                'segformer_config', 'segformer_checkpoint', 'traffic_config', 'traffic_checkpoint'):
        path = Path(cfg[key])
        assets[key] = [str(path), path.stat().st_size, path.stat().st_mtime_ns] if path.exists() else [str(path), None]
    code = hashlib.sha256()
    for p in sorted((ROOT / 'tools/rgb_only').glob('*.py')):
        code.update(p.name.encode())
        code.update(p.read_bytes())
    contract = dict(schema_version=1, config={k: v for k, v in cfg.items()
                    if k not in ('_path','training_output','run_name','num_iters')},
                    input_signature=scene['input_signature'], assets=assets, code_sha256=code.hexdigest(),
                    sensor_inputs=['rgb'], gt_intrinsics_used=False, gt_poses_used=False,
                    gt_depth_used=False, gt_boxes_used=False, gt_masks_used=False)
    path = folder / 'prediction_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise RuntimeError('Prediction config/code/assets changed. Use a new output root to avoid stale mixed stages.')
    write_json(path, contract)
    return signature(contract)


def pose(cfg, folder):
    runtime = ROOT / 'third_party/posefree'
    config = json.loads((runtime / 'configs/default.json').read_text())
    config.update(omega_checkpoint=cfg['omega_checkpoint'], moge_checkpoint=cfg['moge_checkpoint'],
                  output_image_size=cfg['image_size'], window_frames=cfg['window_frames'],
                  window_stride=cfg['window_stride'], window_batch_size=cfg['window_batch_size'],
                  min_overlap_poses=cfg['min_overlap_poses'], min_free_gib_per_gpu=cfg['min_free_gib_per_gpu'])
    write_json(folder / 'pose_config.json', config)
    env = dict(os.environ, PYTHONPATH=str(runtime) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    subprocess.run([sys.executable, '-m', 'adagent_posefree', 'prepare',
                    '--manifest', str(folder / 'pose_manifest.json'), '--config', str(folder / 'pose_config.json'),
                    '--output', str(folder / 'pose'), '--num-gpus', '1'], check=True, env=env)
    cameras = folder / 'pose/scenes' / folder.name / 'cameras.npz'
    table, size = camera_table(cameras, read_scene(folder))
    # Rotate only the world axes to x-forward/y-left/z-up. Camera rays remain OpenCV.
    ordered = sorted(table)
    save_npz(folder / 'predicted_cameras.npz', frame_ids=np.array([k[0] for k in ordered]),
             camera_ids=np.array([k[1] for k in ordered]),
             c2w=np.stack([WORLD_FROM_OPENCV @ table[k][0] for k in ordered]).astype(np.float32),
             K=np.stack([table[k][1] for k in ordered]).astype(np.float32), image_size=np.array(size))


def depth(cfg, folder):
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'third_party/posefree/vendor/moge'))
    from moge.model.v2 import MoGeModel
    if not torch.cuda.is_available():
        raise RuntimeError('MoGe inference requires a CUDA node')
    checkpoint = torch.load(cfg['moge_checkpoint'], map_location='cpu', weights_only=True)
    if not checkpoint['model_config'].get('scale_head'):
        raise ValueError('Metric MoGe-2 checkpoint with a scale head is required')
    model = MoGeModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.eval().to('cuda')
    scene = read_scene(folder)
    table, size = camera_table(folder / 'predicted_cameras.npz', scene)
    for fid, cam, image in records(scene):
        output = folder / 'moge_depth' / (key_name(fid, cam) + '.npz')
        if output.exists():
            with np.load(output, allow_pickle=False) as z:
                if z['depth'].shape == size and z['valid'].shape == size:
                    continue
        with Image.open(image) as im:
            rgb = np.array(im.convert('RGB'), copy=True)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to('cuda')
        K = table[(fid, cam)][1]
        fov = float(np.degrees(2 * np.arctan(size[1] / (2 * K[0, 0]))))
        with torch.inference_mode():
            result = model.infer(tensor, resolution_level=cfg['moge_resolution_level'],
                                 use_fp16=True, apply_mask=False, fov_x=fov)
        d, valid = resized_depth(result['depth'].float().cpu().numpy(),
                                 result['mask'].bool().cpu().numpy(), size)
        valid &= d < cfg['max_depth']
        if not valid.any():
            raise ValueError(f'MoGe predicted no valid depth: {image}')
        save_npz(output, depth=np.where(valid, d, 0), valid=valid,
                 frame_id=np.array(fid), camera_id=np.array(cam), depth_type=np.array('camera_z'),
                 units=np.array('estimated_metres'), fov_source=np.array('predicted_omega_intrinsics'))
        print(f'MOGE_OK {key_name(fid,cam)} valid={valid.mean():.3f}', flush=True)


def sam(cfg, folder):
    from .grounded_sam2 import PersistentGroundedSAM2, stage_frames, validate_masks
    args = SimpleNamespace(gsam_root=Path(cfg['sam_repo']), checkpoint=Path(cfg['sam_checkpoint']),
                           model_cfg=cfg['sam_model_config'], grounding_model=cfg['grounding_model'],
                           text=cfg['sam_text'], step=cfg['sam_detect_interval'])
    model = PersistentGroundedSAM2(args)
    scene = read_scene(folder)
    for cam in scene['camera_ids']:
        frames = [(f, p) for f, c, p in records(scene) if c == cam]
        output = folder / 'sam_tracks' / cam
        if output.exists():
            good, _ = validate_masks(frames, output)
            if good and len(list((output / 'json_data').glob('*.json'))) == len(frames):
                continue
            raise RuntimeError(f'Incomplete SAM output {output}; use a new output root or move this partial camera directory aside')
        # Keep class metadata: it is needed to distinguish humans from vehicles.
        with tempfile.TemporaryDirectory(prefix='adsg_sam_') as temporary:
            staging = Path(temporary) / 'frames'
            stage_frames(frames, staging)
            model.process_video(frames, staging, output)
            # Upstream remaps mask names only. Remap JSON names in a collision-free two-phase pass.
            for i, (fid, _) in enumerate(frames):
                source = output / 'json_data' / f'mask_{i:06d}.json'
                source.rename(output / 'json_data' / f'.mapped_{fid:06d}.json')
            for fid, _ in frames:
                (output / 'json_data' / f'.mapped_{fid:06d}.json').rename(
                    output / 'json_data' / f'mask_{fid:06d}.json')
        good, _ = validate_masks(frames, output)
        if not good:
            raise ValueError(f'SAM coverage is incomplete: {output}')


def semantic(cfg, folder):
    from PIL import Image
    sys.path.insert(0, cfg['segformer_repo'])
    from mmseg.apis import init_segmentor, inference_segmentor
    model = init_segmentor(cfg['segformer_config'], cfg['segformer_checkpoint'], device='cuda:0')
    size = tuple(cfg['image_size'])
    for fid, cam, path in records(read_scene(folder)):
        pred = inference_segmentor(model, str(path))[0]
        if pred.min() < 0 or pred.max() > 18:
            raise ValueError('The configured SegFormer must use the 19-class Cityscapes label contract')
        labels = resized_labels(pred, size)
        for kind, label in [('sky_masks', cfg['sky_class_id']), ('road_masks', cfg['road_class_id'])]:
            out = folder / kind / (key_name(fid,cam) + '.png')
            out.parent.mkdir(exist_ok=True)
            Image.fromarray(((labels == label) * 255).astype(np.uint8)).save(out)


def traffic(cfg, folder):
    sys.path.insert(0, str(ROOT / 'third_party/GroundingDINO'))
    from adsutido_tools.extract_traffic_lights import (build_gdino_model, detect_traffic_lights,
                                                       cluster_detections, triangulate_rays)
    import cv2
    model = build_gdino_model(cfg['traffic_config'], cfg['traffic_checkpoint'], 'cuda')
    scene = read_scene(folder)
    table, size = camera_table(folder / 'predicted_cameras.npz', scene)
    dets = []
    for fid, cam, path in records(scene):
        boxes, scores, rgb = detect_traffic_lights(model, str(path), device='cuda')
        pose, K = table[(fid, cam)]
        with np.load(folder / 'moge_depth' / (key_name(fid,cam) + '.npz')) as z:
            d, valid = z['depth'], z['valid']
        h, w = rgb.shape[:2]
        for box, score in zip(boxes, scores):
            x1,y1,x2,y2 = np.asarray(box) * [size[1]/w, size[0]/h, size[1]/w, size[0]/h]
            xa, ya = max(0,int(x1)), max(0,int(y1))
            xb, yb = min(size[1],int(np.ceil(x2))), min(size[0],int(np.ceil(y2)))
            region = d[ya:yb,xa:xb][valid[ya:yb,xa:xb]]
            if not len(region):
                continue  # no arbitrary constant-depth fallback in the RGB-only branch
            pixel = np.array([(x1+x2)/2,(y1+y2)/2,1])
            ray = np.linalg.inv(K) @ pixel
            xyz = pose[:3,:3] @ (ray * np.percentile(region, 30)) + pose[:3,3]
            direction = pose[:3,:3] @ ray
            direction /= np.linalg.norm(direction)
            dets.append((fid, cam, xyz, [xa,ya,xb,yb], score, pose[:3,3], direction))
    clusters = cluster_detections(dets, eps=cfg['traffic_cluster_eps'], min_samples=cfg['min_track_frames'])
    output = []
    for _, group in sorted(clusters.items()):
        if len({d[0] for d in group}) < cfg['min_track_frames']:
            continue
        point = np.median(np.stack([d[2] for d in group]),axis=0)
        tri, condition = triangulate_rays(np.stack([d[5] for d in group]),np.stack([d[6] for d in group]))
        if tri is not None and condition > 1e-4:
            ranges = [np.dot(tri-d[5],d[6]) for d in group]
            # Keep triangulation only if all observations lie in front and agree with MoGe.
            if min(ranges)>1 and max(ranges)<cfg['max_depth'] and np.linalg.norm(tri-point)<5:
                point = tri
        output.append(dict(center=point.tolist(), observations=[dict(frame_id=d[0],camera_id=d[1],
                           box=d[3],score=float(d[4])) for d in group]))
    write_json(folder / 'traffic_predictions.json', dict(source='existing GroundingDINO + DBSCAN + ray triangulation; MoGe depth',
                                                        instances=output))


def validate(cfg, folder):
    from .assembly import validate_scene
    return validate_scene(folder, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--stage', choices=list(DEPENDENCIES), required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    folder = a.scene_dir.resolve()
    token = claim(cfg, folder)
    for dependency in DEPENDENCIES[a.stage]:
        marker = folder / '.stages' / f'{dependency}.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('signature') != token:
            raise RuntimeError(f'Missing/incompatible prerequisite: {dependency}')
    marker = folder / '.stages' / (a.stage + '.json')
    if marker.exists() and json.loads(marker.read_text()).get('signature') == token and a.stage != 'validate':
        print(f'STAGE_CACHED {a.stage}', flush=True)
        return
    if a.stage == 'assemble':
        from .assembly import assemble
        assemble(cfg, folder)
    else:
        globals()[a.stage](cfg, folder)
    write_json(marker, dict(signature=token, stage=a.stage))
    print(f'STAGE_OK {a.stage}', flush=True)


if __name__ == '__main__':
    main()
