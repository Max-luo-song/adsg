# # 激活 Conda 环境
# CONDA_BASE=$(conda info --base)
# source "${CONDA_BASE}/etc/profile.d/conda.sh"
# conda activate grounddino

# scene_id=621

# python adsutido_tools/extract_traffic_lights.py \
#     --scene_dir /inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed/training/$scene_id \
#     --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#     --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
#     --cam_ids 0 1 2 --save_masks --device cuda

conda activate drivestudio
python adsutido_tools/visualize_traffic_lights.py \
    --output_dir output_adstudio_trafficlight/vis_traffic_lights_621 \
    --cam_ids 0 1 2 \
    --fps 10 \
    --resize_width 960 \
    --max_dist 60