# Pipeline 检查报告

## ✅ 已完成的修复

### 1. 场景目录命名问题
**问题**: NuScenes preprocess.py 输出目录使用3位数字（如`061`），但脚本期望完整场景名（如`scene-0061`）

**解决方案**: 
- 添加 `get_scene_dir_name()` 函数：将 `scene-0061` 转换为 `061`
- 所有脚本统一使用这个转换逻辑

**修改的文件**:
- `pipeline.sh` - 添加转换函数，更新所有验证函数
- `extract_traffic_lights.sh` - 转换场景名到目录名
- `render_novel_trajectory.sh` - 转换GT图像路径

### 2. Marker 文件位置
**修复前**: `/data/nuscenes/processedads/mini/scene_scene-0061/`
**修复后**: `/data/nuscenes/processedads/mini/061/`

### 3. 路径一致性
所有阶段现在都使用相同的目录命名：
```
Preprocess 输出:  /data/.../mini/061/
Traffic Lights:   /data/.../mini/061/instances/
Train 读取:       /data/.../mini/ (data_root) + scene-0061 (scene_idx)
Render GT:        /data/.../mini/061/images/
Markers:          /data/.../mini/061/.xxx_complete
```

## ⚠️ 需要注意的事项

### 1. Train.sh 的 scene_idx 参数
当前 train.sh 传递 `scene-0061` 作为 `data.scene_idx`

**需要确认**: 
- AdStudio 的 config 如何使用 `scene_idx` 参数
- 是期望完整场景名 `scene-0061` 还是数字 `061`

**两种可能**:
```python
# 情况1: config 期望完整场景名（推荐）
scene_dir = os.path.join(data_root, scene_idx)  # mini/scene-0061 ❌

# 情况2: config 需要手动转换
scene_idx_num = convert_scene_name(scene_idx)  # scene-0061 -> 061
scene_dir = os.path.join(data_root, scene_idx_num)  # mini/061 ✅
```

**当前状态**: 脚本传递 `scene-0061`，假设 AdStudio 内部会处理转换

### 2. 建议的测试流程

```bash
# 步骤 1: 测试预处理
cd /path/to/adstudio
./scripts/nuscene/preprocess.sh 61 1 \
  /data/sets/nuscenes \
  /data/nuscenes/processedads \
  v1.0-mini 4 32 0

# 验证输出
ls -la /data/nuscenes/processedads/mini/061/

# 步骤 2: 测试 pipeline（单场景）
./scripts/nuscene/pipeline.sh scene-0061 0 v1.0-mini

# 步骤 3: 检查各阶段 marker
ls -la /data/nuscenes/processedads/mini/061/.*.complete

# 步骤 4: 检查训练输出
ls -la output/nuscenes_scene-0061/
```

## 📋 检查清单

### 预处理阶段
- [ ] 输出目录为 `061` 而非 `scene-0061`
- [ ] 包含 images/, lidar/, calib/, dynamic_masks/ 等目录
- [ ] Marker 文件在正确位置

### 交通灯提取阶段
- [ ] 正确找到场景目录 (`061`)
- [ ] 生成 instances/ 目录
- [ ] Marker 文件创建

### 训练阶段
- [ ] 正确读取数据（data_root + scene_idx）
- [ ] 生成 output/nuscenes_scene-0061/
- [ ] Checkpoint 保存成功
- [ ] Marker 文件创建

### 渲染阶段
- [ ] 找到 checkpoint
- [ ] GT 路径正确（061/images/）
- [ ] 生成 novel_traj/ 输出
- [ ] FID 计算成功

## 🔧 如果出现问题

### 问题: 找不到场景目录
```bash
# 检查实际目录名
ls /data/nuscenes/processedads/mini/

# 应该看到: 000, 001, 061, 等等
# 如果看到: scene-0061, 说明 preprocess.py 行为不同
```

**解决方案**: 如果 preprocess 输出的是完整场景名，需要修改所有脚本移除 `get_scene_dir_name()` 转换

### 问题: 训练无法找到数据
```bash
# 检查 train.py 如何组合路径
# 在 train.sh 中添加 debug 输出
echo "data_root: $processed_dir"
echo "scene_idx: $scene_name"
```

## 📊 当前状态

✅ **已完成**:
- 场景名到目录名的转换逻辑
- Pipeline 各阶段路径统一
- Marker 文件位置修复
- 所有脚本参数传递正确

⚠️ **待验证**:
- Train.sh 的 scene_idx 参数使用方式
- 实际运行测试确认路径正确

## 🎯 总结

Pipeline 脚本现在应该能正确运行，前提是:
1. NuScenes preprocess 输出3位数字目录（如 `061`）
2. AdStudio train.py 接受完整场景名（如 `scene-0061`）作为 scene_idx 并在内部转换

如果以上假设不成立，可能需要微调 train.sh 中的 scene_idx 参数格式。

建议先运行 `validate_pipeline.sh` 查看路径转换逻辑，然后小规模测试单个场景。
