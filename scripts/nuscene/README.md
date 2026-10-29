# NuScenes Scripts for AdStudio

这个目录包含了在 AdStudio 框架下处理 NuScenes 数据集的完整脚本集。

## 文件列表

- `preprocess.sh` - 数据预处理脚本（阶段1-3）
- `extract_traffic_lights.sh` - 交通灯提取脚本
- `train.sh` - 训练脚本
- `render_novel_trajectory.sh` - 渲染新轨迹脚本
- `pipeline.sh` - 完整端到端处理流程
- `batch_pipeline.sh` - 批量处理多个场景
- `scenes.txt` - 场景列表示例文件

## 使用方法

### 1. 单场景预处理

处理单个场景的数据（转换、分割掩码、人体姿态）：

```bash
# 基本用法
./preprocess.sh <start_idx> <num_scenes> [input_dir] [output_dir] [split] [interpolate_N] [workers] [gpu_id]

# 示例：处理 v1.0-mini 中的第一个场景
./preprocess.sh 0 1 \
  /path/to/nuscenes/raw \
  /path/to/output/processedads \
  v1.0-mini 4 32 0
```

**参数说明：**
- `start_idx`: 起始场景索引（0-based）
- `num_scenes`: 要处理的场景数量
- `input_dir`: NuScenes 原始数据目录
- `output_dir`: 输出目录
- `split`: 数据集分割（v1.0-mini, v1.0-trainval 等）
- `interpolate_N`: 帧插值数量（0-4）
- `workers`: 并行工作进程数
- `gpu_id`: GPU ID（0, 1, 2... 或 -1 自动选择）

### 2. 提取交通灯

在预处理后的数据上提取交通灯：

```bash
# 基本用法
./extract_traffic_lights.sh <scene_name> [processed_dir]

# 示例
./extract_traffic_lights.sh scene-0061 /path/to/processedads/mini
```

### 3. 训练模型

在预处理的场景上训练模型：

```bash
# 基本用法
./train.sh <scene_name> [processed_dir] [start_timestep] [end_timestep]

# 示例
./train.sh scene-0061 /path/to/processedads/mini 0 -1
```

**参数说明：**
- `scene_name`: 场景名称（如 scene-0061）
- `processed_dir`: 预处理数据目录
- `start_timestep`: 起始帧索引（默认 0）
- `end_timestep`: 结束帧索引（-1 表示最后一帧）

### 4. 渲染新轨迹

在训练好的模型上渲染新视角：

```bash
# 基本用法
./render_novel_trajectory.sh <scene_name> [processed_dir]

# 示例
./render_novel_trajectory.sh scene-0061 /path/to/processedads/mini
```

此脚本会自动查找最新的checkpoint并渲染多个轨迹类型。

### 5. 完整流程（推荐）

运行完整的端到端流程（预处理→交通灯提取→训练→渲染）：

```bash
# 基本用法
./pipeline.sh <scene_name> [gpu_id] [split]

# 示例
./pipeline.sh scene-0061 0 v1.0-mini
```

**特性：**
- 增量处理：使用 marker 文件跟踪完成状态
- 自动跳过：已完成的阶段会自动跳过
- 数据验证：每个阶段完成后验证输出
- 并发保护：使用文件锁防止冲突
- 详细日志：保存在 `logs/pipeline_nuscene/scene_<name>.log`

### 6. 批量处理

批量处理多个场景，自动管理GPU资源池：

```bash
# 基本用法
./batch_pipeline.sh <scene_list.txt> [gpu_list] [split]

# 示例：使用 GPU 0,1,2,3 处理所有场景
./batch_pipeline.sh scenes.txt "0 1 2 3" v1.0-mini

# 示例：只使用 GPU 0,2
./batch_pipeline.sh scenes.txt "0 2" v1.0-mini
```

**特性：**
- GPU 池调度：自动分配空闲 GPU
- 并行处理：多个场景同时处理
- 失败重试：记录失败场景供后续处理
- 实时监控：彩色输出显示进度
- 日志分离：每个场景独立日志文件

**scenes.txt 格式：**
```
# 每行一个场景名称
scene-0061
scene-0103
scene-0655
# 支持注释和空行
```

## 数据目录结构

### 输入（原始 NuScenes 数据）
```
/path/to/nuscenes/raw/
├── samples/          # 关键帧图像
├── sweeps/           # 中间帧
├── v1.0-mini/        # 分割元数据
└── ...
```

### 输出（预处理后）
```
/path/to/processedads/
└── mini/                    # split_short (去掉 v1.0- 前缀)
    └── scene-0061/          # 场景目录
        ├── images/          # 相机图像
        ├── lidar/           # 点云数据
        ├── calib/           # 标定参数
        ├── dynamic_masks/   # 动态物体掩码
        ├── road_masks/      # 道路分割
        ├── sky_masks/       # 天空分割
        ├── humanpose/       # 人体姿态
        └── instances/       # 交通灯实例
```

## 配置选项

### 摄像头配置

默认使用前3个摄像头：
- 0: CAM_FRONT
- 1: CAM_FRONT_LEFT  
- 2: CAM_FRONT_RIGHT

如需修改，编辑 `train.sh` 中的 `camera_ids` 数组。

### 轨迹类型

`render_novel_trajectory.sh` 中可配置渲染的轨迹：
- `left_shift_1m` / `left_shift_3m` - 左移轨迹
- `right_shift_1m` / `right_shift_3m` - 右移轨迹
- `front_shift_*` - 前移轨迹
- `back_shift_*` - 后移轨迹
- `change_lane_*` - 变道轨迹

### 配置文件

训练使用的配置文件（在 `train.sh` 中修改）：
```bash
config_file="configs/omnire_extended_cam_lidar_nuscenes.yaml"
dataset_config="nuscenes/3cams"
```

## 环境要求

需要以下 conda 环境：
- `drivestudio` - 数据预处理和训练
- `segformer` - 语义分割
- `4D-humans` - 人体姿态估计
- `grounddino` - 交通灯检测（可选）

## 故障排除

### 问题：找不到 checkpoint
确保训练已完成，checkpoint 保存在 `output/nuscenes_<scene_name>/*/checkpoints/` 目录下。

### 问题：SMPL 提取失败
确保已切换到 AdStudio 目录，PYTHONPATH 包含 AdStudio，以访问 `third_party/Humans4D` 和 `third_party/smplx` 子模块。

### 问题：GPU 内存不足
减少 batch size 或降低图像分辨率。

### 问题：预处理被跳过
删除 marker 文件重新运行：
```bash
rm -f /path/to/processedads/mini/scene_<name>/.preprocess_complete
```

## 与 KITTI 脚本的区别

| 特性 | KITTI | NuScenes |
|------|-------|----------|
| 场景标识 | `2011_09_26_drive_0001_sync` | `scene-0061` |
| 数据组织 | 按日期分组 | 按split分组 (mini/trainval) |
| 摄像头 | 2个（左右） | 6个（默认使用前3个） |
| 帧插值 | 不支持 | 支持 (interpolate_N) |
| 配置文件 | `omnire_extended_cam_lidar_kitti.yaml` | `omnire_extended_cam_lidar_nuscenes.yaml` |

## 示例工作流

```bash
# 1. 处理单个场景的完整流程
cd /path/to/adstudio
./scripts/nuscene/pipeline.sh scene-0061 0 v1.0-mini

# 2. 批量处理多个场景
./scripts/nuscene/batch_pipeline.sh scripts/nuscene/scenes.txt "0 1 2 3" v1.0-mini

# 3. 单独重新渲染（如果训练完成但渲染失败）
./scripts/nuscene/render_novel_trajectory.sh scene-0061 /path/to/processedads/mini
```

## 日志位置

- Pipeline 日志: `logs/pipeline_nuscene/scene_<name>.log`
- Batch 日志: `logs/batch_nuscene/pipeline_<scene_name>.log`

## 维护者

适配自 KITTI 脚本，针对 NuScenes 数据集修改。
