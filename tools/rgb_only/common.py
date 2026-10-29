"""RGB-only contracts and geometry. No Waymo calibration/label reader lives here."""
import hashlib
import json
from pathlib import Path

import numpy as np

WORLD_FROM_OPENCV = np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                              [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def read_scene(folder):
    return json.loads((Path(folder) / 'rgb_scene.json').read_text())


def records(scene):
    for frame in scene['frames']:
        for camera in scene['camera_ids']:
            yield frame['frame_id'], str(camera), Path(frame['images'][str(camera)])


def key_name(frame, camera):
    return f'{frame:03d}_{camera}'


def camera_table(path, scene):
    with np.load(path, allow_pickle=False) as z:
        ids, cams = z['frame_ids'], z['camera_ids'].astype(str)
        poses, intrinsics = z['c2w'].astype(np.float64), z['K'].astype(np.float64)
        size = tuple(int(x) for x in z['image_size'])
    expected = {(f, c) for f, c, _ in records(scene)}
    keys = list(zip(ids.tolist(), cams.tolist()))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Camera cache must cover exactly the indexed RGB images')
    if poses.shape != (len(keys), 4, 4) or intrinsics.shape != (len(keys), 3, 3):
        raise ValueError('Invalid predicted camera shape')
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError('Nonfinite predicted camera values')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]) or not np.allclose(intrinsics[:, 2], [0, 0, 1]):
        raise ValueError('Invalid homogeneous camera matrices')
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-4):
        raise ValueError('Predicted camera rotation is not orthogonal')
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-4):
        raise ValueError('Predicted camera rotation is reflected')
    if (intrinsics[:, (0, 1), (0, 1)] <= 0).any() or len(size) != 2 or min(size) < 1:
        raise ValueError('Invalid focal length/image grid')
    return {key: (poses[i], intrinsics[i]) for i, key in enumerate(keys)}, size


def unproject(depth, K, c2w, valid=None, stride=1):
    """Depth is camera z, in estimated metres; local camera axes stay OpenCV."""
    h, w = depth.shape
    y, x = np.mgrid[:h:stride, :w:stride]
    z = depth[y, x]
    keep = np.isfinite(z) & (z > 0)
    if valid is not None:
        keep &= valid[y, x]
    y, x, z = y[keep], x[keep], z[keep]
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T
    camera = rays * z[:, None]
    return camera @ c2w[:3, :3].T + c2w[:3, 3], y, x


def resized_depth(depth, valid, size):
    import cv2
    valid = valid & np.isfinite(depth) & (depth > 0)
    target = (size[1], size[0])
    weights = cv2.resize(valid.astype(np.float32), target, interpolation=cv2.INTER_LINEAR)
    values = cv2.resize(np.where(valid, depth, 0).astype(np.float32), target,
                        interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(valid.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST).astype(bool)
    out = values / np.maximum(weights, 1e-8)
    mask &= (weights > 0.5) & np.isfinite(out) & (out > 0)
    return np.where(mask, out, 0).astype(np.float32), mask


def resized_labels(labels, size):
    import cv2
    return cv2.resize(labels.astype(np.int32), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def merge_camera_tracks(observations, threshold=1.5, min_shared=3):
    """Conservative cross-camera association; never merge distinct same-camera IDs."""
    keys = sorted(observations)
    parent = {key: key for key in keys}
    members = {key: {key[0]} for key in keys}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    candidates = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0] or observations[a]['class_name'] != observations[b]['class_name']:
                continue
            shared = sorted(set(observations[a]['centers']) & set(observations[b]['centers']))
            if len(shared) < min_shared:
                continue
            distance = np.median([np.linalg.norm(observations[a]['centers'][f] -
                                               observations[b]['centers'][f]) for f in shared])
            if distance < threshold:
                candidates.append((distance, a, b))
    for _, a, b in sorted(candidates):
        a, b = root(a), root(b)
        if a != b and not (members[a] & members[b]):
            parent[b] = a
            members[a] |= members[b]
    return {key: root(key) for key in keys}
