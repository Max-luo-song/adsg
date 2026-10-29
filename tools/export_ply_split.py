import argparse
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
from plyfile import PlyData, PlyElement


CATEGORY_TO_MODEL_KEY: Dict[str, str] = {
    "background": "Background",
    "rigid": "RigidNodes",
    "pedestrian": "SMPLNodes",
    "traffic_light": "TrafficLightNodes",
    "road": "RoadNodes",
}
DEFAULT_TARGETS: List[str] = [
    "background",
    "rigid",
    "pedestrian",
    "traffic_light",
    "road",
]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Gaussian:
    def __init__(self, gaussian_state: Dict[str, torch.Tensor]):
        self._means = _to_numpy(gaussian_state["_means"])
        self._features_dc = _to_numpy(gaussian_state["_features_dc"])
        if "_features_rest" in gaussian_state and gaussian_state["_features_rest"] is not None:
            self._features_rest = _to_numpy(gaussian_state["_features_rest"])
        else:
            self._features_rest = np.zeros((self._means.shape[0], 0, 3), dtype=np.float32)
        self._opacities = _to_numpy(gaussian_state["_opacities"])
        self._scales = _to_numpy(gaussian_state["_scales"])
        self._quats = _to_numpy(gaussian_state["_quats"])


def _construct_attribute_names(num_f_rest: int, scale_dim: int, quat_dim: int) -> List[str]:
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attrs.append(f"f_dc_{i}")
    for i in range(num_f_rest):
        attrs.append(f"f_rest_{i}")
    attrs.append("opacity")
    for i in range(scale_dim):
        attrs.append(f"scale_{i}")
    for i in range(quat_dim):
        attrs.append(f"rot_{i}")
    return attrs


def write_gaussian_to_ply(gaussian: Gaussian, out_path: str) -> None:
    xyz = gaussian._means
    normals = np.zeros_like(xyz)
    f_dc = gaussian._features_dc.reshape((gaussian._features_dc.shape[0], -1))
    f_rest = gaussian._features_rest.reshape((gaussian._features_rest.shape[0], -1))
    opacities = gaussian._opacities.reshape((gaussian._opacities.shape[0], -1))
    scales = gaussian._scales
    rotations = gaussian._quats

    num_f_rest = f_rest.shape[1]
    names = _construct_attribute_names(num_f_rest, scales.shape[1], rotations.shape[1])
    dtype_full = [(name, "f4") for name in names]

    attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scales, rotations], axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def export_split_ply(
    pth_path: str,
    out_dir: str,
    targets: List[str],
    skip_missing: bool,
) -> None:
    ckpt = torch.load(pth_path, map_location="cpu")
    if "models" not in ckpt or not isinstance(ckpt["models"], dict):
        raise ValueError(f"Checkpoint does not contain a valid 'models' dict: {pth_path}")
    models = ckpt["models"]

    os.makedirs(out_dir, exist_ok=True)
    for category in targets:
        model_key = CATEGORY_TO_MODEL_KEY[category]
        if model_key not in models:
            msg = f"Model key '{model_key}' not found in checkpoint."
            if skip_missing:
                print(f"[Skip] {category}: {msg}")
                continue
            raise KeyError(msg)

        out_path = osp.join(out_dir, f"{category}.ply")
        write_gaussian_to_ply(Gaussian(models[model_key]), out_path)
        print(f"[OK] {category} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export gaussians by category to separate ply files")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir. Default: <model_dir>/point_clouds_split",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        choices=DEFAULT_TARGETS,
        default=DEFAULT_TARGETS,
        help="Categories to export",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip categories whose model key is absent in checkpoint",
    )
    args = parser.parse_args()

    run_dir = osp.dirname(args.model)
    out_dir = args.out_dir if args.out_dir is not None else osp.join(run_dir, "point_clouds_split")
    export_split_ply(args.model, out_dir, args.targets, args.skip_missing)
