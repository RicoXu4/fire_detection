# 火灾检测 AI 系统

这是一个面向部署的火焰目标检测系统，使用 YOLO 系列检测模型权重完成图像输入、算法识别、标签输出和可视化展示。工程默认兼容 Ultralytics YOLO 权重格式，可直接替换为训练好的 `YOLO v26` 火灾检测权重。

## 功能

- 图像文件 CLI 推理，输出 JSON 标签和可视化图片
- FastAPI 云端推理接口，支持上传图片并返回检测结果
- 可配置置信度、IoU、输入尺寸、设备和类别过滤
- 训练配置模板，支持通过额外火灾数据集继续训练提升精度
- Dockerfile，便于云服务器、容器平台和 Kubernetes 部署

## 目录

```text
fire_detection/
  app/
    api.py              # 云端 HTTP 推理服务
    config.py           # 配置项
    detector.py         # YOLO 模型封装
    schemas.py          # 输出结构
    utils.py            # 图像读写和可视化
  configs/
    fire_dataset.yaml   # 训练数据集配置模板
  scripts/
    infer.py            # 本地图片推理
    train.py            # 训练入口
  Dockerfile
  requirements.txt
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

准备模型权重。将训练好的 YOLO v26 火焰检测权重放到 `weights/yolov26_fire.pt`，或通过环境变量指定：

```bash
export FIRE_MODEL_PATH=/path/to/yolov26_fire.pt
```

本地推理：

```bash
python scripts/infer.py \
  --model weights/yolov26_fire.pt \
  --source data/test.jpg \
  --output runs/predict \
  --conf 0.25
```

启动云端服务：

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

调用接口：

```bash
curl -X POST "http://127.0.0.1:8000/detect?visualize=true" \
  -F "file=@data/test.jpg"
```

返回内容包含检测框、类别标签、置信度、图片尺寸和可视化结果的 base64 PNG。

## 训练数据集格式

推荐使用 YOLO 标注格式：

```text
datasets/fire/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

每个标签文件一行：

```text
class_id x_center y_center width height
```

坐标均为 0-1 归一化值。类别建议：

- `0`: fire
- `1`: smoke

## 实验数据集来源

本项目实验中使用或评估过的数据集如下：

| 数据集 | 来源项目地址 | 本地路径 | 用途 |
| --- | --- | --- | --- |
| Fire Smoke Obstacle Dataset v10 | <https://universe.roboflow.com/myworkspace-d5kq3/fire-smoke-obstacle-dataset> | `datasets/mydata_fire` | 原始 fire/smoke 训练、验证和测试数据集 |
| Home Fire Dataset | <https://www.kaggle.com/datasets/pengbo00/home-fire-dataset> | `external_datasets/home_fire_dataset/.kagglehub_cache/datasets/pengbo00/home-fire-dataset/versions/1` | 家庭/室内火灾场景继续微调 |
| D-Fire / Smoke Fire Detection YOLO | <https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo> | `external_datasets/dfire_test_only` | 外部测试和可视化评估 |
| Fire-focused 数据集 | 项目内二次构建，无独立外部地址 | `datasets/mydata_fire_focused` | 由原始 Fire Smoke Obstacle 数据集筛选生成，减少 smoke-only 训练样本 |
| Mixed Indoor 数据集 | 项目内二次构建，无独立外部地址 | `datasets/mixed_indoor_fire_70_30` | 由 Fire-focused 数据集与 Home Fire Dataset 按比例混合生成 |

其中 D-Fire 在本项目中主要作为外部评估数据使用，未作为主要训练集参与训练。相关评估输出位于 `runs/detect/runs/eval/dfire_*`。

启动训练：

```bash
python scripts/train.py \
  --model weights/yolov26.pt \
  --data configs/fire_dataset.yaml \
  --epochs 100 \
  --imgsz 640
```

训练建议：

- 合并公开火焰数据、自采监控场景和夜间/逆光/小目标样本
- 加入难负样本，例如灯光、夕阳、反光、红色物体、焊接火花
- 保留独立验证集，按场景划分而不是随机混合
- 部署前按业务阈值评估漏检率和误报率

## Google Colab 训练

Colab 入口 notebook：

```text
notebooks/fire_risk_colab_training.ipynb
```

如果要把本地单类 `fire_risk` 数据集带到 Colab，先生成一个不含断链的 zip：

```bash
python scripts/package_colab_dataset.py \
  --dataset datasets/mixed_indoor_fire_risk_plus_new \
  --output fire_detection_datasets.zip
```

把 `fire_detection_datasets.zip` 上传到 Google Drive 的 `MyDrive`，在 Colab 中打开 notebook 后按顺序运行。默认训练配置为单类 `fire_risk`、`imgsz=768`、`60` epochs，并把训练结果复制回 Google Drive。

## Docker 部署

```bash
docker build -t fire-detection:latest .
docker run --rm -p 8000:8000 \
  -e FIRE_MODEL_PATH=/models/yolov26_fire.pt \
  -v /path/to/weights:/models \
  fire-detection:latest
```

## 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FIRE_MODEL_PATH` | `weights/yolov26_fire.pt` | 模型权重路径 |
| `FIRE_CONF_THRESHOLD` | `0.25` | 置信度阈值 |
| `FIRE_IOU_THRESHOLD` | `0.45` | NMS IoU 阈值 |
| `FIRE_IMAGE_SIZE` | `640` | 推理输入尺寸 |
| `FIRE_DEVICE` | `cpu` | `cpu`、`cuda` 或 `0` |
| `FIRE_CLASSES` | `fire,flame,smoke` | 保留的类别名，留空表示不过滤 |
