# 训练脚本包快速开始

本包仅包含训练、预处理、测试及负样本下载所需代码和环境配置。数据集、Python/ONNX 模型权重、预训练缓存、训练日志、网页资源和虚拟环境均不包含在内。完整项目说明见 README.md；其中涉及网页发布、历史权重或项目维护的命令需要另行克隆源码仓库或自行提供相应资源。

## 1. 解压两个包

维护者分别提供：

- GitHub Release：`naifamily-training-kit.zip`，脚本包；
- 维护者的其他存储位置：`naifamily-dataset-v1.2.zip`，数据包。

把两者解压到同一个父目录，合并同名 `naifamily-training-kit/` 目录。数据包不会覆盖脚本。完成后的目录应为：

```text
naifamily-training-kit/
  model.py
  train.py
  test.py
  setup_env.ps1
  scripts/train_face_rejection.py
  data_processed_v12/manifest.json
  data_processed_v12/train/...
  data_processed_v12/val/...
  data_processed_v12/test/...
  hard_negatives/mixed_v12/manifest.json
  hard_negatives/mixed_v12/lfw/...
  hard_negatives/mixed_v12/cartoonset/...
```

数据包含 448 张处理后奶家人图片（382/33/33）及 1,200 张真实/卡通负样本（840/180/180），并保留来源和第三方归属说明。已经是 RGB 224×224 的正样本不必再次预处理。原始负样本由训练脚本按配置处理。

## 2. 创建环境

安装 Python 3.12，在上述目录中运行：

```powershell
.\setup_env.ps1 -Backend xpu
# NVIDIA 显卡改为 -Backend cuda；CPU 改为 -Backend cpu
```

首次安装依赖和首次下载 torchvision 官方 ResNet18 初始化权重需要联网。脚本包不包含可直接复用的 .venv。

## 3. 从头训练，再加入负样本微调

本包不提供历史 best.pt，按下列流程自行训练，避免引用不存在的历史权重：

```powershell
# 第一阶段：ImageNet 初始化，学习 11 类奶家人
.\.venv\Scripts\python.exe train.py --data data_processed_v12 --output runs\local_base --device auto

# 第二阶段：加入真实/卡通人物负样本，保留第一阶段的验证/测试划分
.\.venv\Scripts\python.exe scripts\train_face_rejection.py --data data_processed_v12 --weights runs\local_base\best.pt --negatives hard_negatives\mixed_v12 --output runs\local_oe --device auto --epochs 30 --lr 0.00003 --head-lr 0.0003

# 测试最终模型
.\.venv\Scripts\python.exe test.py --data data_processed_v12 --weights runs\local_oe\best.pt --output test_results\local_oe --device auto
```

所有输出目录必须选择新的空目录。第一阶段会保存 best.pt 和 loss 曲线；第二阶段保存 best.pt、history.json、loss_curve.png、evaluation.json，后者含已知类别及真实/卡通负样本的保留测试结果。此处从头训练的结果可能与 README 中维护者从历史权重继续训练的结果不同。

只运行 train.py 不包含未知负样本训练；如需保留人物拒判能力，应完成第二阶段。

## 4. 使用自己的图片

用 `preprocess.py --source <按类别组织的原图目录> --output <新的处理后目录>` 创建数据，再在上述命令中用 `--data` 指定新目录。本包的负样本微调脚本面向含“豆奶”标签的奶家人任务；其他类别体系可使用通用 train.py，负样本脚本的类别专项评估也需相应调整。每批共同参考图派生的图片应保留来源分组，避免跨训练与验证/测试集合造成泄漏。

只有在未获得负样本数据时，才需要运行 `scripts/prepare_face_negatives.py` 和 `scripts/prepare_cartoon_negatives.py` 联网获取公开负样本。下载后的路径为 `hard_negatives/mixed_v12/`，第三方图片按各自来源许可使用。
