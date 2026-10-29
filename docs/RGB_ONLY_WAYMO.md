# Waymo RGB-only per-scene reconstruction

Implementation lives in `tools/rgb_only`, `datasets/waymo/rgb_*`, and
`models/trainers/rgb_scene_graph.py`. Use the dedicated config
`configs/omnire_rgb_only.yaml`; the original LiDAR pipeline remains available.

## Data and scene selection

`data/waymo_rgb_scenes.csv` copies the eight entries and inclusive frame ranges
from Omniscenegraph's `data/waymo_example_scenes.txt`. The top-level `scenes.txt`
contains nuScenes names, so it is deliberately not used for Waymo.

Only `<rgb_root>/<numeric_scene>/images/<frame>_<camera>.jpg` is read from the
source dataset. No source intrinsics, extrinsics, ego poses, LiDAR, object boxes,
SMPL, road masks, or dynamic masks are copied into the prediction dataset.
Original frame IDs are retained, including scene 172's nonzero first frame.
Output paths are separate from source data, and image files are symlinked.

## Execution

Edit `configs/rgb_only/waymo.json` for runtime executables/GPU/assets if needed.
Use the project root as the working directory:

```bash
bash scripts/waymo/rgb_pipeline.sh --stage plan
bash scripts/waymo/rgb_pipeline.sh --stage doctor --scenes 621
bash scripts/waymo/rgb_pipeline.sh --stage index --scenes 621
bash scripts/waymo/rgb_pipeline.sh --stage all --scenes 621
```

Run a single stage with `--stage pose|depth|sam|semantic|traffic|assemble|validate|train`.
All stages require their prerequisite completion markers. `all` runs preflight
before launching any inference. Logs are under the prediction scene's `logs/`.
The JSON config maps stages to independent Python executables, allowing the
existing MoGe, Grounded-SAM, SegFormer, and ADSG training environments to coexist.
No models are downloaded automatically. Preflight reports missing assets,
imports and CUDA instead of substituting GT data or guessed calibration.

After one scene succeeds, remove `--scenes 621` to process all eight reference
scenes sequentially. Missing source scenes must be provided as RGB before they
can run; a selected subset never silently changes the reference split.

The training run directory must not exist. Use a new `run_name` to repeat a run.
Changing prediction configuration, inputs, assets or preprocessing code requires
a new prediction output root. Per-stage markers are published only on success.
Completed compatible stages are reused; interrupted SAM camera outputs are
retained and explicitly reported for review rather than silently overwritten.

## Predicted data contract

1. **Cameras:** migrated standalone VGGT-Omega + MoGe-2/GCA metric-scale recovery
   and overlapping-window SE(3) alignment. `predicted_cameras.npz` stores per-frame
   pinhole `K`, OpenCV camera-to-world transforms, frame/camera IDs and `[H,W]`.
   The world basis is rotated once to approximately x-forward/y-left/z-up.
   Camera-local axes stay right/down/forward. No original lens distortion is used.
2. **Depth:** MoGe-2 metric camera-z predictions, using Omega's estimated FoV;
   resized with valid-pixel weights to the same direct-resize grid as K. Invalid
   depth is zero and has a separate boolean mask. Estimated metres are not
   measured/GPS scale.
3. **Objects:** copied Grounded-SAM2 persistent video tracking and class metadata.
   Per-camera IDs are associated conservatively using simultaneous 3D agreement;
   distinct IDs from one camera cannot merge. Predicted depth lifts visible
   surfaces; robust trajectory span distinguishes moving objects from parked
   detections. Coarse yaw/boxes initialize RigidNodes and DeformableNodes.
   Humans use DeformableNodes, since no RGB-only SMPL estimator was requested.
4. **Road/sky:** SegFormer Cityscapes predictions (road=0, sky=10), newly generated
   on the RGB inputs. Road points initialize RoadNodes without a LiDAR ground fit.
5. **Traffic lights:** reuse ADSG's GroundingDINO detection, DBSCAN clustering,
   and ray triangulation. MoGe supplies the depth prior. Unsupported depth pixels
   are skipped; the old fixed-depth/LiDAR fallback is not used.
6. **Initialization:** seed positions/colors are unprojected from RGB and MoGe.
   Moving object points are transformed into object coordinates using their
   same-frame predicted poses. Road, background and instance seeds are separate.
   Held-out frame seeds are excluded from Gaussian initialization. Dense MoGe
   depth enters the existing depth-loss tensor slot; no LiDAR source is created.

`rgb_scene.json`, `prediction_contract.json`, `assembly.json`, and `validation.json`
record inputs, provenance and coverage. Validation checks every indexed image,
camera, depth/mask grid, instance ID, and seed attribute before training.

## Evaluation and limits

This is offline per-scene reconstruction: all supplied RGB frames may be used
to estimate camera poses and track identities. Held-out RGB therefore does enter
the camera/tracking preprocessing; do not describe this as a causal/online or
strict held-out-image geometry benchmark. Gaussian initialization excludes
held-out depth samples, but instance trajectories still use offline predictions.

Grounded-SAM gives 2D masks, not 3D object pose. The lifting, cross-camera
association, motion threshold, yaw, and box sizes are geometric bootstraps and
need scene-level inspection. Monocular depth scale error, partial visible
surfaces, occlusions, camera drift and ID switches can affect dynamic objects.
Signal-light depth and small-object seeds may be sparse. The initial world basis
uses camera orientation; it does not estimate measured gravity or GPS position.

GPU inference and actual training must be verified on a CUDA node with compatible
ADSG dependencies. CPU contract tests and source indexing do not establish
rendering quality or GPU runtime compatibility. See the saved doctor report and
implementation status for the validation actually performed.

## Source provenance

- `third_party/posefree`: copied from
  `yx-ufo/e2_front352_20260930/source/geometry/posefree`, including its backend,
  vendored model code, license notices and tests. Large checkpoints stay in their
  existing shared location.
- `tools/rgb_only/grounded_sam2.py`: copied from that source project's
  `tools/r9/preprocess_waymo_sam2_tracks.py`. The wrapper keeps the original
  detection/forward/reverse tracking algorithm and retains class JSON metadata.
  Its mask utility is copied into `tools/rgb_only/sam_masks.py`; only the utility
  import is redirected to avoid collision with ADSG's `utils` package.
- Traffic-light detector/triangulation functions and ADSG rendering, node models,
  losses and optimizer are reused from this project.

Run CPU contract tests with:

```bash
/root/miniconda3/envs/dggt/bin/python -m unittest discover -s tests -p 'test_rgb_only.py' -v
```
