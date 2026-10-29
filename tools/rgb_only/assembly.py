"""Lift tracked 2D instances with predicted cameras/depth; build training artifacts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (camera_table, key_name, merge_camera_tracks, read_scene, records,
                     resized_labels, save_npz, unproject, write_json)


def class_name(label):
    label = label.lower().strip(' .')
    if label in ('person', 'pedestrian', 'human'):
        return 'Pedestrian'
    if label in ('bicycle', 'motorcycle', 'cyclist'):
        return 'Cyclist'
    if label in ('car', 'truck', 'bus', 'vehicle'):
        return 'Vehicle'
    raise ValueError(f'Unsupported Grounded-SAM class: {label!r}')


def view(folder, fid, cam, size):
    key = key_name(fid, cam)
    with np.load(folder / 'moge_depth' / (key + '.npz'), allow_pickle=False) as z:
        depth, valid = z['depth'].copy(), z['valid'].copy().astype(bool)
    if depth.shape != size or valid.shape != size:
        raise ValueError(f'Depth grid mismatch: {key}')
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError(f'Invalid depth values: {key}')
    mask_path = folder / 'sam_tracks' / cam / 'mask_data' / f'mask_{fid:06d}.npy'
    labels = np.load(mask_path, allow_pickle=False)
    if labels.ndim != 2 or (labels < 0).any() or not np.equal(labels, labels.astype(np.int64)).all():
        raise ValueError(f'Invalid SAM instance map: {mask_path}')
    labels = resized_labels(labels, size)
    meta_path = folder / 'sam_tracks' / cam / 'json_data' / f'mask_{fid:06d}.json'
    classes = json.loads(meta_path.read_text())['labels']
    for label in np.unique(labels):
        if label and str(int(label)) not in classes:
            raise ValueError(f'Missing class metadata for {key} instance {label}')
    road = np.array(Image.open(folder / 'road_masks' / (key + '.png')).convert('L')) > 0
    sky = np.array(Image.open(folder / 'sky_masks' / (key + '.png')).convert('L')) > 0
    if road.shape != size or sky.shape != size:
        raise ValueError(f'Semantic mask grid mismatch: {key}')
    return depth, valid & ~sky, labels, classes, road, sky


def assemble(cfg, folder):
    folder = Path(folder)
    scene = read_scene(folder)
    table, size = camera_table(folder / 'predicted_cameras.npz', scene)
    tracks = {}
    # First pass: sparse, robust 3D track observations, without reading Waymo labels.
    for fid, cam, _ in records(scene):
        depth, valid, labels, classes, _, _ = view(folder, fid, cam, size)
        points, y, x = unproject(depth, table[(fid,cam)][1], table[(fid,cam)][0], valid, stride=2)
        local_ids = labels[y,x]
        for local in np.unique(local_ids):
            if local == 0:
                continue
            points_local = points[local_ids == local]
            if len(points_local) < cfg['min_object_pixels']:
                continue
            lo, hi = np.percentile(points_local, [10,90], axis=0)
            key = (cam,int(local))
            canonical = class_name(classes[str(int(local))]['class_name'])
            track = tracks.setdefault(key, dict(class_name=canonical, centers={}, sizes=[]))
            if track['class_name'] != canonical:
                raise ValueError(f'Inconsistent class in tracked object {key}')
            track['centers'][fid] = (lo+hi)/2
            track['sizes'].append(np.maximum(hi-lo, 0.1))
    aliases = merge_camera_tracks(tracks, cfg['cross_camera_distance'], cfg['min_track_frames'])
    groups = {}
    for key, parent in aliases.items():
        groups.setdefault(parent, []).append(key)
    instances, active_lookup = {}, {}
    for parent, members in sorted(groups.items()):
        frame_centers = {}
        for member in members:
            for fid, center in tracks[member]['centers'].items():
                frame_centers.setdefault(fid, []).append(center)
        fids = sorted(frame_centers)
        if len(fids) < cfg['min_track_frames']:
            continue
        centers = np.stack([np.median(np.stack(frame_centers[f]),axis=0) for f in fids])
        # Temporal medians suppress single-frame monocular-depth spikes.
        smooth = np.stack([np.median(centers[max(0,i-1):i+2],axis=0) for i in range(len(centers))])
        span = np.linalg.norm(np.percentile(smooth,90,axis=0)-np.percentile(smooth,10,axis=0))
        if span < cfg['moving_threshold_m']:
            continue  # a detectable vehicle is not necessarily a moving vehicle
        idx = len(instances)
        sizes = np.stack([s for k in members for s in tracks[k]['sizes']])
        extent = np.maximum(np.percentile(sizes,75,axis=0), [0.2,0.2,0.2])
        poses = []
        for i, center in enumerate(smooth):
            motion = smooth[min(i+1,len(smooth)-1)]-smooth[max(i-1,0)]
            yaw = np.arctan2(motion[1],motion[0]) if np.linalg.norm(motion[:2])>0.05 else 0.0
            co, si = np.cos(yaw), np.sin(yaw)
            pose = np.eye(4)
            pose[:3,:3] = [[co,-si,0],[si,co,0],[0,0,1]]
            pose[:3,3] = center
            poses.append(pose.tolist())
        # The coarse box is a bootstrap from visible surfaces, not a foundation-model 3D box prediction.
        horizontal = max(extent[0],extent[1])
        extent[:2] = horizontal
        instances[str(idx)] = dict(id=f'grounded_sam_{idx}', class_name=tracks[parent]['class_name'],
            source_tracks=[list(m) for m in members], moving_span_m=float(span),
            frame_annotations=dict(frame_idx=fids,obj_to_world=poses,box_size=[extent.tolist() for _ in fids]))
        for member in members:
            active_lookup[member] = idx
    frame_ids = {f['frame_id'] for f in scene['frames']}
    traffic = json.loads((folder/'traffic_predictions.json').read_text())['instances']
    tl_by_view = {}
    for item in traffic:
        fids = sorted({o['frame_id'] for o in item['observations']})
        if not set(fids) <= frame_ids:
            raise ValueError('Traffic predictions contain frames outside the RGB manifest')
        idx = len(instances)
        pose = np.eye(4)
        pose[:3,3] = item['center']
        instances[str(idx)] = dict(id=f'traffic_{idx}',class_name='TrafficLight',
            frame_annotations=dict(frame_idx=fids,obj_to_world=[pose.tolist() for _ in fids],
                                   box_size=[[0.6,0.6,1.0] for _ in fids]))
        for obs in item['observations']:
            key = (obs['frame_id'],str(obs['camera_id']))
            if key not in table:
                raise ValueError('Traffic prediction refers to an unknown camera')
            tl_by_view.setdefault(key, []).append((idx,obs['box']))
    frame_instances = {str(f): [] for f in sorted(frame_ids)}
    for idx, inst in instances.items():
        for fid in inst['frame_annotations']['frame_idx']:
            frame_instances[str(fid)].append(int(idx))
    write_json(folder/'instances/instances_info.json',instances)
    write_json(folder/'instances/frame_instances.json',frame_instances)
    rng = np.random.default_rng(17)
    chunks = {k: [] for k in ('points','colors','frame_ids','instance_ids','road')}
    for fid,cam,image in records(scene):
        depth,valid,labels,_,road,sky = view(folder,fid,cam,size)
        ids = np.full(size,-1,np.int32)
        for local in np.unique(labels):
            idx = active_lookup.get((cam,int(local)))
            if local and idx is not None and idx in frame_instances[str(fid)]:
                ids[labels==local] = idx
        for idx,box in tl_by_view.get((fid,cam),[]):
            x1,y1,x2,y2 = box
            if not (0<=x1<=x2<=size[1] and 0<=y1<=y2<=size[0]):
                raise ValueError('Traffic box outside image grid')
            region = ids[y1:y2,x1:x2]
            region[region<0] = idx
        dynamic = ids>=0
        human = np.zeros(size,bool)
        vehicle = np.zeros(size,bool)
        for idx in np.unique(ids[dynamic]):
            name = instances[str(idx)]['class_name']
            if name in ('Pedestrian','Cyclist'):
                human |= ids==idx
            elif name=='Vehicle':
                vehicle |= ids==idx
        moving = human|vehicle
        for kind,mask in [('all',moving),('human',human),('vehicle',vehicle)]:
            out=folder/'dynamic_masks'/kind/(key_name(fid,cam)+'.png')
            out.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray((mask*255).astype(np.uint8)).save(out)
        save_npz(folder/'instance_masks'/(key_name(fid,cam)+'.npz'),ids=ids)
        # Dense depth uses its own grid; seed sampling bounds training RAM independently.
        points,y,x=unproject(depth,table[(fid,cam)][1],table[(fid,cam)][0],valid,stride=cfg['point_stride'])
        object_ids=ids[y,x]
        take=[]
        for idx in np.unique(object_ids):
            pool=np.flatnonzero(object_ids==idx)
            budget=cfg['points_per_image'] if idx<0 else cfg['points_per_object_view']
            take.append(rng.choice(pool,min(len(pool),budget),replace=False))
        take=np.concatenate(take) if take else np.empty(0,int)
        if len(take):
            rgb=np.asarray(Image.open(image).convert('RGB').resize((size[1],size[0]),Image.Resampling.BILINEAR))
            chunks['points'].append(points[take].astype(np.float32))
            chunks['colors'].append(rgb[y[take],x[take]].astype(np.float32)/255)
            chunks['frame_ids'].append(np.full(len(take),fid,np.int32))
            chunks['instance_ids'].append(object_ids[take])
            chunks['road'].append(road[y[take],x[take]] & (object_ids[take]<0))
    if not chunks['points']:
        raise ValueError('No valid RGB-predicted seed points')
    save_npz(folder/'seed_points.npz',**{k:np.concatenate(v) for k,v in chunks.items()})
    write_json(folder/'assembly.json',dict(schema_version=1,input_signature=scene['input_signature'],
        sensor_inputs=['rgb'],image_size=list(size),num_instances=len(instances),
        depth_source='MoGe-2',camera_source='VGGT-Omega + MoGe/GCA + overlap SE3',
        object_source='Grounded-SAM2 masks lifted by predicted depth; coarse motion/box bootstrap',
        road_source='SegFormer Cityscapes',traffic_source='existing detector and clustering with RGB-predicted geometry',
        coordinate_frame='initial front camera rotated to x-forward/y-left/z-up; estimated metres',
        gt_intrinsics_used=False,gt_poses_used=False,gt_depth_used=False,gt_boxes_used=False,gt_masks_used=False))
    validate_scene(folder,cfg)


def validate_scene(folder,cfg):
    folder=Path(folder)
    scene=read_scene(folder)
    _,size=camera_table(folder/'predicted_cameras.npz',scene)
    meta=json.loads((folder/'assembly.json').read_text())
    if meta['input_signature']!=scene['input_signature'] or meta['sensor_inputs']!=['rgb']:
        raise ValueError('Invalid RGB-only assembly provenance')
    if any(meta.get(k) is not False for k in ('gt_intrinsics_used','gt_poses_used','gt_depth_used','gt_boxes_used','gt_masks_used')):
        raise ValueError('Ground-truth fields are forbidden')
    instances=json.loads((folder/'instances/instances_info.json').read_text())
    frame_instances=json.loads((folder/'instances/frame_instances.json').read_text())
    if set(instances)!={str(i) for i in range(len(instances))}:
        raise ValueError('Instance IDs must be contiguous')
    expected_frames={f['frame_id'] for f in scene['frames']}
    if set(frame_instances)!={str(f) for f in expected_frames}:
        raise ValueError('Frame-instance map coverage mismatch')
    for idx,inst in instances.items():
        ann=inst['frame_annotations']
        if inst['class_name'] not in ('Vehicle','Pedestrian','Cyclist','TrafficLight'):
            raise ValueError('Invalid predicted instance class')
        fids=ann['frame_idx']
        poses=np.asarray(ann['obj_to_world']); sizes=np.asarray(ann['box_size'])
        if len(set(fids))!=len(fids) or not set(fids)<=expected_frames or poses.shape!=(len(fids),4,4) or sizes.shape!=(len(fids),3):
            raise ValueError('Invalid instance observations')
        if not np.isfinite(poses).all() or not np.isfinite(sizes).all() or (sizes<=0).any():
            raise ValueError('Invalid instance geometry')
        for fid in fids:
            if int(idx) not in frame_instances[str(fid)]:
                raise ValueError('Inconsistent frame-instance map')
    for fid,cam,_ in records(scene):
        view(folder,fid,cam,size)
        for kind in ('all','human','vehicle'):
            mask=np.array(Image.open(folder/'dynamic_masks'/kind/(key_name(fid,cam)+'.png')))
            if mask.shape!=size or not np.isin(mask,[0,255]).all():
                raise ValueError('Invalid dynamic mask grid/values')
        with np.load(folder/'instance_masks'/(key_name(fid,cam)+'.npz')) as z:
            ids=z['ids']
            if ids.shape!=size or not set(np.unique(ids))<=({-1}|set(int(x) for x in instances)):
                raise ValueError('Invalid instance ID map')
    with np.load(folder/'seed_points.npz',allow_pickle=False) as z:
        n=len(z['points'])
        if n<4 or z['points'].shape!=(n,3) or z['colors'].shape!=(n,3):
            raise ValueError('Insufficient/invalid seed points')
        if not np.isfinite(z['points']).all() or not np.isfinite(z['colors']).all() or ((z['colors']<0)|(z['colors']>1)).any():
            raise ValueError('Invalid seed coordinates/colors')
        if any(z[k].shape!=(n,) for k in ('frame_ids','instance_ids','road')):
            raise ValueError('Seed attribute shape mismatch')
        if not set(z['frame_ids'].tolist())<=expected_frames:
            raise ValueError('Seed frames outside RGB manifest')
        if not set(z['instance_ids'].tolist())<=({-1}|set(int(x) for x in instances)):
            raise ValueError('Seed IDs outside instance contract')
        background=(z['instance_ids']<0)&~z['road']
        road=(z['instance_ids']<0)&z['road']
        if background.sum()<4 or road.sum()<4:
            raise ValueError('Need at least four predicted points in both background and road')
    report=dict(ok=True,scene=scene['scene_name'],frames=len(scene['frames']),cameras=scene['camera_ids'],
                image_size=list(size),seed_points=n,instances=len(instances),rgb_only=True)
    write_json(folder/'validation.json',report)
    print(json.dumps(report),flush=True)
    return report
