# 基于 YOLO26n 的火灾图像检测实验报告

## 摘要

本实验围绕火灾图像中的明火与烟雾目标检测任务，构建了一个从数据准备、模型训练、指标评估到 Web 推理部署的完整火灾检测系统。项目采用 Ultralytics YOLO 兼容的 YOLO26n 检测模型作为基础模型，检测类别包括 `fire` 与 `smoke`。实验中先使用原始火灾数据集进行基线训练，再针对烟雾样本占比和室内火灾场景泛化问题，分别构建 fire-focused 数据集、Home Fire 微调数据集与 mixed indoor 混合数据集，并比较不同训练方案的精度、召回率与 mAP 指标。最终系统提供命令行推理、FastAPI 图像检测接口和前端可视化页面，可输出检测框、类别、置信度和检测结果图。

## 1. 实验目的

本实验的主要目标如下：

1. 掌握 YOLO 单阶段目标检测模型在火灾检测任务中的训练与推理流程。
2. 构建适用于火灾检测的 YOLO 格式数据集，完成 `fire` 与 `smoke` 两类目标检测。
3. 通过数据筛选、数据混合和继续微调，提高模型在明火、烟雾和室内火灾场景下的识别能力。
4. 使用 Precision、Recall、mAP50 和 mAP50-95 等指标评价模型效果。
5. 将训练得到的模型封装为本地 CLI 与 Web API，形成可部署的火灾检测应用。

## 2. 实验环境

本项目主要实验环境如下：

| 项目 | 配置 |
| --- | --- |
| 操作系统 | macOS |
| 深度学习框架 | PyTorch / Ultralytics |
| 训练设备 | Apple MPS、CPU |
| 模型 | YOLO26n |
| 输入尺寸 | 256、320、416、512 |
| 主要依赖 | `ultralytics`、`fastapi`、`uvicorn`、`opencv-python`、`pydantic` |
| 服务形式 | CLI 推理、FastAPI HTTP 接口、静态前端页面 |

项目入口包括：

- 训练脚本：`scripts/train.py`
- 推理脚本：`scripts/infer.py`
- Web API：`app/api.py`
- 模型封装：`app/detector.py`
- 前端页面：`app/static/index.html`
- 模型设计说明：`docs/model_design.md`

## 3. 数据集准备

### 3.1 数据格式

实验使用 YOLO 标注格式，每张图片对应一个 `.txt` 标签文件。每行标签格式如下：

```text
class_id x_center y_center width height
```

其中坐标均为 0 到 1 之间的归一化值。本实验设置两个类别：

| 类别编号 | 类别名 | 含义 |
| --- | --- | --- |
| 0 | fire | 明火、火舌、燃烧区域 |
| 1 | smoke | 烟雾 |

### 3.2 数据集来源

本实验使用的数据集来源如下：

| 数据集 | 来源 | 项目地址 | 项目中位置 | 说明 |
| --- | --- | --- | --- | --- |
| 原始 Fire/Smoke 数据集 | Roboflow Universe: Fire Smoke Obstacle Dataset v10 Fire_Smoke Detection | <https://universe.roboflow.com/myworkspace-d5kq3/fire-smoke-obstacle-dataset> | `datasets/mydata_fire` | 数据集自带 README 显示其于 2025 年 3 月 25 日从 Roboflow 导出，包含 11027 张图片，标注格式为 YOLOv8；项目中将其作为基础 fire/smoke 数据集。 |
| Home Fire Dataset | KaggleHub 缓存数据集 `pengbo00/home-fire-dataset` | <https://www.kaggle.com/datasets/pengbo00/home-fire-dataset> | `external_datasets/home_fire_dataset/.kagglehub_cache/datasets/pengbo00/home-fire-dataset/versions/1` | 用于家庭/室内火灾场景继续微调，提高模型对室内明火场景的适应能力。 |
| D-Fire 测试数据集 | KaggleHub 缓存数据集 `sayedgamal99/smoke-fire-detection-yolo` | <https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo> | `external_datasets/dfire_test_only` | 本项目中主要用于外部测试和可视化评估，没有参与主要训练。 |
| Fire-focused 数据集 | 项目内二次构建 | 无独立外部项目地址 | `datasets/mydata_fire_focused` | 由 `scripts/prepare_fire_focused_dataset.py` 从原始 Fire/Smoke 数据集中筛选生成，不是独立外部数据集。 |
| Mixed Indoor 数据集 | 项目内二次构建 | 无独立外部项目地址 | `datasets/mixed_indoor_fire_70_30` | 由 `scripts/prepare_mixed_indoor_dataset.py` 将 Fire-focused 数据集与 Home Fire Dataset 按比例混合生成。 |

原始数据集配置文件为 `configs/mydata_fire.yaml`，目录划分如下：

| 数据集划分 | 图片数 | 标签数 |
| --- | ---: | ---: |
| train | 8826 | 8826 |
| valid | 1088 | 1088 |
| test | 1107 | 1107 |

### 3.3 Fire-focused 数据集

Fire-focused 数据集来源于原始 Roboflow Fire Smoke Obstacle 数据集，是项目内为了调整类别分布而生成的派生数据集。在原始数据集中，烟雾样本可能对明火检测造成一定干扰。为提高明火类检测效果，项目编写了 `scripts/prepare_fire_focused_dataset.py`，按照以下策略构建 fire-focused 数据集：

1. 对训练集优先保留包含 `fire` 类的图片。
2. 对仅包含 `smoke` 类的训练图片只保留一部分，默认保留比例为 35%。
3. 验证集和测试集保持原始划分，以便与原始数据集结果对比。
4. 使用软链接方式构建新数据集，避免重复复制图片。

构建后的 `datasets/mydata_fire_focused` 数据集规模如下：

| 数据集划分 | 图片数 | 标签数 |
| --- | ---: | ---: |
| train | 6643 | 6643 |
| valid | 1088 | 1088 |
| test | 1107 | 1107 |

### 3.4 Mixed Indoor 数据集

为了提高模型对室内家庭火灾场景的泛化能力，项目继续引入 KaggleHub 缓存的 Home Fire Dataset，并编写 `scripts/prepare_mixed_indoor_dataset.py` 构建混合数据集。该脚本将 fire-focused 数据集与 Home Fire 数据集混合，默认训练集 Home Fire 样本占比约为 30%，验证集 Home Fire 样本占比约为 50%。构建后的数据集目录为 `datasets/mixed_indoor_fire_70_30`。

混合数据集规模如下：

| 数据集划分 | 图片数 | 标签数 |
| --- | ---: | ---: |
| train | 9490 | 9490 |
| valid | 2176 | 2176 |
| test | 2407 | 2407 |

### 3.5 D-Fire 外部测试集

项目中还保留了 D-Fire 相关测试数据，来源为 KaggleHub 缓存数据集 `sayedgamal99/smoke-fire-detection-yolo`。该数据集在本实验中主要用于训练后外部测试和样例可视化，而不是作为主要训练集参与模型训练。

本项目为 D-Fire 构建了两个测试视图：

| 测试视图 | 配置文件 | 图片数 | 标签数 | 用途 |
| --- | --- | ---: | ---: | --- |
| Fire/Smoke 顺序测试集 | `external_datasets/dfire_test_only/dfire_test_fire_smoke_order.yaml` | 4306 | 4306 | 按 `fire=0`、`smoke=1` 的类别顺序进行完整外部测试 |
| Near-fire 子集 | `external_datasets/dfire_test_only/dfire_near_fire_test.yaml` | 636 | 636 | 选取接近火灾场景的子集，用于观察模型在相似但跨数据源场景下的泛化表现 |

对应的评估输出保存在 `runs/detect/runs/eval/dfire_*` 目录中，例如 `dfire_test_best_fire_smoke_order`、`dfire_near_fire_test_best` 和 `dfire_sample_best_conf020`。

## 4. 模型与方法

### 4.1 模型结构

实验采用 YOLO26n 作为基础模型。YOLO 属于单阶段目标检测模型，其基本流程为：

1. 输入图像经过缩放、归一化等预处理。
2. Backbone 提取颜色、纹理、边缘和上下文特征。
3. Neck 融合多尺度特征，提高小目标和远距离火焰检测能力。
4. Head 输出目标类别、置信度和边界框。
5. 使用 NMS 根据置信度和 IoU 阈值过滤重复检测框。

训练后的 YOLO26n 模型约 237.5 万参数，计算量约 5.2 GFLOPs，适合在轻量化部署场景中使用。

### 4.2 训练策略

训练入口为 `scripts/train.py`。主要训练策略包括：

- 使用预训练 YOLO26n 权重进行迁移学习。
- 使用余弦学习率衰减 `cos_lr=True`。
- 使用 Mosaic、随机缩放、平移、水平翻转和 HSV 色彩扰动等数据增强。
- 训练后期关闭 Mosaic，以稳定边界框定位。
- 使用 MPS 设备完成主要训练，CPU 仅用于快速 smoke test。
- 针对 Apple MPS 训练稳定性，部分正式实验关闭 AMP。

关键增强参数如下：

| 参数 | 数值 | 作用 |
| --- | ---: | --- |
| `hsv_h` | 0.015 | 色调扰动 |
| `hsv_s` | 0.7 | 饱和度扰动 |
| `hsv_v` | 0.4 | 亮度扰动 |
| `degrees` | 5.0 | 小角度旋转 |
| `translate` | 0.1 | 平移增强 |
| `scale` | 0.5 | 缩放增强 |
| `fliplr` | 0.5 | 水平翻转 |
| `close_mosaic` | 8 或 10 | 训练后期关闭 Mosaic |

### 4.3 实验方案

本项目实际完成了以下几组训练：

| 实验名称 | 数据集 | 初始权重 | epoch | imgsz | batch | device | 说明 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| CPU smoke fast | `configs/mydata_fire.yaml` | `yolo26n.pt` | 1 | 256 | 1 | CPU | 快速验证训练流程 |
| 原始数据集训练 | `configs/mydata_fire_abs.yaml` | `yolo26n.pt` | 20 | 416 | 4 | MPS | 原始 fire/smoke 数据集基线 |
| Fire-focused 训练 | `datasets/mydata_fire_focused/data.yaml` | `yolo26n.pt` | 40 | 416 | 4 | MPS | 减少 smoke-only 训练样本，提高明火关注度 |
| Home Fire 微调 | `external_datasets/home_fire_dataset/home_fire_dataset.yaml` | fire-focused best | 25 | 416 | 4 | MPS | 基于家庭火灾数据集继续微调 |
| Mixed Indoor 训练 | `datasets/mixed_indoor_fire_70_30/data.yaml` | fire-focused best | 35 | 512 | 4 | MPS | 混合原始与室内火灾样本，提升场景泛化 |

## 5. 评价指标

实验使用目标检测常用指标评价模型：

| 指标 | 含义 |
| --- | --- |
| Precision | 检测为火灾目标的结果中，真正正确的比例 |
| Recall | 所有真实火灾目标中，被模型检出的比例 |
| mAP50 | IoU 阈值为 0.5 时的平均精度 |
| mAP50-95 | IoU 从 0.5 到 0.95 多阈值下的平均精度，更严格 |

在火灾检测任务中，漏报通常比误报代价更高，因此 Recall 是非常重要的指标；但实际部署中也需要控制误报，否则会影响报警系统可用性。

## 6. 实验结果与分析

### 6.1 训练结果对比

各实验最后一轮验证指标如下：

| 实验名称 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| CPU smoke fast | 0.0028 | 0.3754 | 0.0013 | 0.0004 |
| 原始数据集训练 20e | 0.6214 | 0.4682 | 0.5289 | 0.2739 |
| Fire-focused 40e | 0.6348 | 0.4795 | 0.5513 | 0.2905 |
| Home Fire 微调 25e | **0.8743** | **0.8125** | **0.8836** | **0.5335** |
| Mixed Indoor 35e | 0.7498 | 0.5905 | 0.6828 | 0.3807 |

从结果可以看出：

1. CPU smoke fast 只训练 1 个 epoch，主要用于验证训练链路，不具备实际检测效果。
2. 原始数据集训练 20 轮后，模型已经具备基础的火焰和烟雾检测能力，mAP50 达到 0.5289。
3. Fire-focused 数据集通过减少 smoke-only 样本，使 mAP50 从 0.5289 提升到 0.5513，说明面向明火目标的数据筛选有一定收益。
4. Home Fire 微调效果最好，Precision、Recall 和 mAP 均明显提升，mAP50 达到 0.8836，说明针对家庭火灾场景继续微调可以显著提升同类场景检测效果。
5. Mixed Indoor 训练的 mAP50 为 0.6828，低于 Home Fire 单独微调，但高于原始数据集和 fire-focused 数据集，说明混合数据有助于提高场景覆盖，但不同数据源混合后任务难度也会增加。

### 6.2 Fire-focused 模型分类别结果

Fire-focused 模型在验证集上的最终分类别结果如下：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fire | 670 | 1240 | 0.642 | 0.537 | 0.619 | 0.334 |
| smoke | 701 | 1255 | 0.627 | 0.422 | 0.485 | 0.247 |
| all | 1088 | 2495 | 0.635 | 0.480 | 0.552 | 0.290 |

该结果表明，模型对 `fire` 类的检测效果优于 `smoke` 类。原因可能是明火具有更明显的颜色和边缘特征，而烟雾形态更分散、边界模糊，且容易与雾气、云、反光等背景混淆。

### 6.3 可视化结果

项目保存了训练曲线、PR 曲线、混淆矩阵和样例检测图。典型输出包括：

- 训练曲线：`runs/detect/runs/train/yolo26n_home_fire_finetune_mps_25e/results.png`
- 混淆矩阵：`runs/detect/runs/train/yolo26n_home_fire_finetune_mps_25e/confusion_matrix.png`
- PR 曲线：`runs/detect/runs/train/yolo26n_home_fire_finetune_mps_25e/BoxPR_curve.png`
- 样例预测图：`runs/detect/runs/predict/sample_test_best_conf020/preview_grid.jpg`
- D-Fire 样例预测图：`runs/detect/runs/eval/dfire_sample_best_conf020/preview_grid_predictions.jpg`

这些可视化结果可以用于观察模型是否存在漏检、误检以及检测框定位不准等问题。

## 7. 系统实现

### 7.1 本地推理

项目提供 `scripts/infer.py` 进行单张图片推理。推理流程为：

1. 加载训练得到的 `.pt` 权重。
2. 读取输入图片。
3. 调用 `FireDetector.predict()` 执行 YOLO 推理。
4. 保存 JSON 检测结果和可视化图片。

示例命令：

```bash
python scripts/infer.py \
  --model runs/detect/runs/train/yolo26n_home_fire_finetune_mps_25e/weights/best.pt \
  --source runs/predict/sample_inputs/flare_0222_jpg.rf.68cd4ea058cbf49c796b730d160aa8d1.jpg \
  --output runs/predict \
  --conf 0.25 \
  --device mps
```

### 7.2 Web API

项目使用 FastAPI 实现云端推理接口，核心文件为 `app/api.py`。主要接口如下：

| 接口 | 方法 | 功能 |
| --- | --- | --- |
| `/` | GET | 返回前端页面 |
| `/health` | GET | 健康检查 |
| `/detect` | POST | 上传图片并返回检测结果 |

`/detect` 接口支持以下参数：

- `visualize`：是否返回可视化图片。
- `conf`：整体置信度阈值。
- `fire_conf`：明火类别置信度阈值。
- `smoke_conf`：烟雾类别置信度阈值。
- `label_mode`：标签显示模式，支持 `raw` 与 `risk`。

返回结果包含图片尺寸、检测框、类别、置信度和可选的 base64 可视化图片。

### 7.3 前端演示

项目还实现了静态前端页面，支持：

1. 点击或拖拽上传图片。
2. 分别调节 fire 与 smoke 类别置信度阈值。
3. 调用 `/detect` 接口进行识别。
4. 展示原图、检测后图片、检测数量、图片尺寸和检测结果表格。

这使模型不仅能在命令行中运行，也能以 Web 服务形式完成简单交互演示。

### 7.4 部署方式

项目提供 Dockerfile，可将模型服务打包为容器。部署时通过环境变量指定权重路径、置信度阈值、IoU 阈值和推理设备：

```bash
docker build -t fire-detection:latest .
docker run --rm -p 8000:8000 \
  -e FIRE_MODEL_PATH=/models/yolov26_fire.pt \
  -v /path/to/weights:/models \
  fire-detection:latest
```

## 8. 问题与改进方向

实验中仍存在以下不足：

1. 烟雾类检测召回率相对较低。烟雾边界模糊、形态变化大，后续可以增加烟雾专门数据集和难负样本。
2. 数据源之间存在分布差异。Home Fire 微调指标最高，但可能更偏向家庭火灾场景；Mixed Indoor 更接近多场景泛化，但指标相对下降。
3. 目前主要使用图片级测试，若面向监控部署，还需要增加视频流测试和持续误报率统计。
4. 火灾报警业务更关注漏检成本，后续应根据实际场景设置更细的 fire/smoke 分类阈值。
5. 可以加入灯光、夕阳、车灯、焊接火花、红色广告牌等难负样本，以降低误报。

后续改进方向包括：

- 扩充夜间、低照度、远距离小火点和遮挡火焰样本。
- 单独构建难负样本验证集，统计误报类型。
- 尝试更大模型或蒸馏模型，对比精度与推理速度。
- 对模型进行 ONNX 或 TensorRT 导出，提升部署性能。
- 在 Web 端增加批量检测、历史记录和报警阈值配置功能。

## 9. 结论

本实验完成了一个基于 YOLO26n 的火灾目标检测系统。通过原始数据集训练、fire-focused 数据筛选、Home Fire 数据集微调和 mixed indoor 混合训练，验证了数据策略对火灾检测模型性能的影响。实验结果表明，直接使用原始数据集训练可获得基础检测能力；减少 smoke-only 样本后，模型在 fire/smoke 总体检测上略有提升；在 Home Fire 数据集上继续微调后，模型取得最佳结果，Precision 为 0.8743，Recall 为 0.8125，mAP50 为 0.8836，mAP50-95 为 0.5335。

除模型训练外，项目还实现了 CLI 推理、FastAPI 服务、前端可视化和 Docker 部署配置，形成了从实验训练到应用部署的完整流程。整体来看，本项目能够完成图片火灾风险检测，并具备进一步扩展到监控场景和云端报警系统的基础。
