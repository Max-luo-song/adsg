# ADSG — RGB-only Waymo scene reconstruction

RGB-only foundation-model preprocessing and per-scene scene graph training. Start with [the RGB-only Waymo guide](docs/RGB_ONLY_WAYMO.md).

Pipeline: VGGT-Omega + MoGe/GCA cameras, MoGe depth, Grounded-SAM2 tracks, SegFormer road/sky, and the existing traffic-light detector/triangulation.

**Validation status:** 8 new CPU tests and 18 migrated camera tests pass; real GPU inference and training are pending. See [implementation status](docs/RGB_ONLY_IMPLEMENTATION_STATUS.json).

Run from the repository root after configuring the local asset/runtime paths:

```bash
bash scripts/waymo/rgb_pipeline.sh --stage doctor --scenes 621
bash scripts/waymo/rgb_pipeline.sh --stage all --scenes 621
```

## Existing ADSG / ADstudio documentation

# README for ADstudio

## 版本时间线
v1 commit：20260318

## 版本描述
v1: 红绿灯+路面(无环境删除策略)+刚体分离

## 关键文件

## 数据
- 数据目录是：`/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo`
- 源数据会下载到：`/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/raw`
- 处理好的数据：`/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed/training{scene_id}`

## 训练

### 脚本使用
- `scripts/waymo/pipeline.sh scene_id`
  用于单个场景执行 adstudio, 结果会保存在 output 中
- `scripts/waymo/batch_pipeline.sh scenes.txt`
  用于多场景同时执行 adstudio，结果会保存在 output 中

- `scripts/waymo/drivestudio_pipeline.sh scene_id`
  用于单个场景执行 drivestudio，结果会保存到 drivestudio/output
- `scripts/waymo/batch_drivestudio_pipeline.sh scenes.txt`
  用于多场景同时执行 drivestudio，结果会保存在 drivestudio/output

### Pipeline 流程
对于 pipeline，整个流程包括：
1. 源数据下载(包含 google 验证过程) —— `scripts/waymo/waymo_download.sh`
2. 数据预处理 —— `scripts/waymo/convert_data.sh`
3. 语义分割 —— `scripts/waymo/semantic_segment.sh`
4. smpl 行人处理 —— `scripts/waymo/extract_human_pose.sh`
5. 红绿灯检测处理 —— `scripts/waymo/extract_traffic_lights.sh`
6. 模型训练 —— `scripts/waymo/train.sh`
7. 新轨迹数据生成 —— `scripts/waymo/render_novel_trajectory.sh`
8. 计算新轨迹 fid —— `scripts/waymo/cal_video_fid.sh`

### 工具程序
- `scripts/waymo/preprocess.sh`
  是整个预处理，但是为了方便监控，分成三部分执行了 (暂时无效)
- `scripts/waymo/visualize_traffic_lights.sh`
  可视化红绿灯的 bounding box 框，结果会存储在 `output_adstudio_trafficlight`
- `scripts/stitch_videos.sh`
  用于把新轨迹生成的视频拼接到一起

## 日志
日志文件夹是：`logs`

里面的结构如下：
- `logs/batch`: adstudio 批量运行结果
- `logs/batch_drivestudio`: drivestudio 批量运行结果
- `logs/drivestudio_pipeline`: drivestudio 单个运行结果
- `logs/pipeline`: adstudio 单个运行结果

# README for Drivestudio
三个shell文件
- train.sh 训练的
- eval.sh 评估的
- cal_video_fid.sh 算fid的
都是项目目录下直接bash