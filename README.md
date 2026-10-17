# 奶家人识别器

基于 PyTorch CNN 的 11 类图像分类项目，提供模型训练、测试、单图推理和浏览器端识别。默认网络为 ImageNet 预训练 ResNet18，同时保留从零训练的轻量 CNN。

- 在线演示：[奶家人识别器](https://bdlfy114514.github.io/naifamily-classifier/)
- 源代码：[bdlfy114514/naifamily-classifier](https://github.com/bdlfy114514/naifamily-classifier)
- 分类标签：奶史、奶狗、奶猫、奶粉、奶蛋、奶蛙、奶鸡、奶鼠、奶龙、小奶、豆奶。

## 1. 获取项目

### 1.1 源代码仓库

```powershell
git clone https://github.com/bdlfy114514/naifamily-classifier.git
cd naifamily-classifier
```

仓库包含代码、静态网页、展示图、浏览器模型及依赖清单。训练数据、Python 最优权重、虚拟环境和训练缓存不随 Git 仓库提交。直接克隆仓库后，如需训练，应准备自己的数据并按第 3 节预处理，或使用下述完整训练包。

### 1.2 本地完整训练包

文件名：`naifamily-training-kit.zip`。该压缩包在维护者本地 `dist/` 目录生成，不上传到 Git 仓库。

训练包包含：

- 全部源代码、本文档、依赖清单和环境配置脚本；
- `data_processed/`：229 张已处理图片及固定划分，train/val/test 分别为 163/33/33；
- `runs/resnet18_transfer_v1/`：最优 Python 权重、训练配置、日志及 loss 曲线；
- `.cache/torch/checkpoints/resnet18-f37072fd.pth`：预训练初始化权重，避免训练时再次下载；
- 对应测试报告、网页文件、展示图和 ONNX 浏览器模型；
- `KIT_MANIFEST.json`：包内各文件的大小和 SHA-256，用于完整性核验。

解压后进入 `naifamily-training-kit/`，在该目录执行下文命令。不要对包内 `data_processed/` 再次裁剪或缩放。最初的原始训练图片已删除，包内保留的是可直接用于训练的处理后数据。

压缩包不包含 Python 安装程序、显卡驱动、依赖安装包或 `.venv`。这些环境文件不可跨电脑直接复用；首次配置依赖需要联网。安装完成后，包内数据及初始化权重可用于本地训练。

## 2. 配置运行环境

### 2.1 环境要求

- Python 3.12，64 位；
- PyTorch 2.9.1、torchvision 0.24.1；
- NumPy、Pillow、Matplotlib，版本见 `requirements.txt`；
- GPU 为可选项。CUDA 需要兼容的 NVIDIA 显卡和驱动，XPU 需要兼容的 Intel 显卡和驱动；
- Node.js 仅用于网页回归测试，不是训练或 Python 测试的必需依赖。

### 2.2 Windows：自动创建 venv

先安装 Python 3.12，并确认以下命令可用：

```powershell
py -3.12 --version
```

在项目根目录选择一种后端安装：

```powershell
# 仅 CPU
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Backend cpu

# NVIDIA CUDA；默认使用 cu128，也可指定 cu126 或 cu130
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Backend cuda -CudaWheel cu128

# Intel XPU
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Backend xpu
```

同一个 venv 一次使用一种 PyTorch 后端。切换后端时重新执行对应命令。CUDA wheel 版本须与显卡及驱动兼容，不能仅根据电脑是否安装 CUDA Toolkit 判断。

如果 Python 未注册到 `py` 启动器，可显式指定解释器：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Backend cpu -Python 'C:\Python312\python.exe'
```

脚本会创建 `.venv/`、安装依赖并执行 `pip check`。已有 `.venv` 会复用其解释器；若解释器版本不正确，请先更换环境。下文直接调用 venv 内的 Python，不要求先激活环境。

### 2.3 Linux：手动创建环境

以下以 CPU 为例：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip check
```

CUDA/XPU 安装时，应选择 PyTorch 官方提供、与当前操作系统和硬件兼容的 wheel。固定版本安装入口见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。Linux 用户执行后文命令时，将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`，并相应调整路径分隔符。

`requirements-lock.txt` 是维护者 XPU 环境的完整版本快照，不应作为跨平台、跨后端的统一安装入口；优先使用上述方式安装。

### 2.4 验证环境

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('XPU:', hasattr(torch, 'xpu') and torch.xpu.is_available())"
.\.venv\Scripts\python.exe -m pip check
```

`--device auto` 按 CUDA、XPU、CPU 的顺序选择可用后端。显式指定不可用的后端会报错。

## 3. 准备数据

### 3.1 使用训练包内的数据

确认根目录中存在 `data_processed/manifest.json` 和 `train/`、`val/`、`test/` 子目录，即可直接进入第 4 节。训练和测试会验证图片尺寸、像素哈希、类别映射与划分信息。

当前数据集由项目最初提供的角色图片经预处理得到，原始类别文件夹名用作标签。现存的 229 张处理后图片按类别统计如下；表中顺序同时是当前模型的输出索引顺序：

| 输出索引 | 类别 | 训练集 | 验证集 | 测试集 | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 奶史 | 15 | 3 | 3 | 21 |
| 1 | 奶狗 | 14 | 3 | 3 | 20 |
| 2 | 奶猫 | 14 | 3 | 3 | 20 |
| 3 | 奶粉 | 15 | 3 | 3 | 21 |
| 4 | 奶蛋 | 14 | 3 | 3 | 20 |
| 5 | 奶蛙 | 16 | 3 | 3 | 22 |
| 6 | 奶鸡 | 15 | 3 | 3 | 21 |
| 7 | 奶鼠 | 14 | 3 | 3 | 20 |
| 8 | 奶龙 | 14 | 3 | 3 | 20 |
| 9 | 小奶 | 16 | 3 | 3 | 22 |
| 10 | 豆奶 | 16 | 3 | 3 | 22 |
| 合计 | 11 类 | 163 | 33 | 33 | 229 |

数据和标签的组织方式：

- **任务类型**：单标签、整图分类，每张图片只有一个类别。没有目标边界框、分割掩码或“非奶家人”负类；网页的低匹配提示由后处理阈值实现。
- **图片格式**：统一保存为 224×224 的 RGB PNG。原图经方向校正、透明背景合成、边缘裁剪，再等比例缩放并补白；训练时读取为 `(3, 224, 224)` 张量。
- **输入归一化**：RGB 值先除以 255，再逐通道减去 mean `(0.485, 0.456, 0.406)`、除以 std `(0.229, 0.224, 0.225)`，与预训练主干的输入约定一致。
- **划分方法**：固定种子 42，各类别分别随机划分，验证/测试目标比例各 15%，按类别四舍五入。实际每类验证和测试均为 3 张，其余进入训练，因此总数并非严格的 70%/15%/15%。
- **数据清单**：`manifest.json` 记录类别映射、原始来源相对路径、处理后路径、划分及像素 SHA-256。当前批次未发现完全重复副本；像素哈希不能自动排除不同截图或轻微改动形成的近重复样本。
- **增强范围**：只在读取训练集时进行水平翻转、轻量旋转、平移、缩放及颜色扰动，不额外增加清单中的样本数。验证和测试保持固定输入。

`show/` 用作网页展示和交互示例，不作为额外的独立评测集，也不重复计入上述 229 张统计。原始未处理图片已删除，训练包保留处理后图片；仅克隆源码仓库不会获得这批训练数据。

### 3.2 使用自己的图片

按类别整理原始图片，例如：

```text
raw_images/
  奶蛙/
    image001.png
    image002.jpg
  奶龙/
    image001.png
    image002.jpg
  ...
```

至少需要两个类别，每类去除完全重复图后至少 5 张。类别文件夹名即输出标签。不要把 `show/` 展示图目录当作训练集。

先检查裁剪配置并生成预览：

```powershell
.\.venv\Scripts\python.exe preprocess.py --source raw_images --output preview_custom --preview-only
```

查看 `preview_custom/preview.jpg`，必要时修改 `preprocess_config.json`。裁剪比例依次为 `[左, 上, 右, 下]`，按文件配置优先于按类别配置，最后使用默认配置。当前默认裁去底部 6%；奶史额外裁去顶部 8.5%。这些参数针对本项目数据，并不适用于所有新图片。

确认预览后输出新数据集：

```powershell
.\.venv\Scripts\python.exe preprocess.py --source raw_images --output data_custom --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

脚本执行 EXIF 方向校正、透明背景合成、裁边、等比例缩放补边至 224×224、完全重复图去重，以及按类别随机划分。原图不覆盖，输出目录必须为空。轻微改动的近重复图片需人工检查，避免进入不同集合。

## 4. 训练模型

### 4.1 网络架构

默认结构由 `model.py` 中的 `build_model(arch='resnet18')` 构建：加载 torchvision 的 `ResNet18_Weights.IMAGENET1K_V1` 初始化主干，将原来的 1000 类分类层替换为 `Dropout(0.2) → Linear(512, 11)`。分类数会根据训练数据清单自动适配；以下尺寸和参数量以当前 11 类、224×224 输入为例。

| 阶段 | 主要结构 | 单张图片的输出尺寸（省略批次维） |
| --- | --- | --- |
| 输入 | RGB、归一化 | `3 × 224 × 224` |
| stem | 7×7 卷积，64 通道，stride 2；BatchNorm；ReLU | `64 × 112 × 112` |
| maxpool | 3×3 最大池化，stride 2 | `64 × 56 × 56` |
| layer1 | 2 个 BasicBlock，64 通道 | `64 × 56 × 56` |
| layer2 | 2 个 BasicBlock，128 通道，首块下采样 | `128 × 28 × 28` |
| layer3 | 2 个 BasicBlock，256 通道，首块下采样 | `256 × 14 × 14` |
| layer4 | 2 个 BasicBlock，512 通道，首块下采样 | `512 × 7 × 7` |
| avgpool / flatten | 自适应全局平均池化至 1×1，再展平 | `512` |
| 分类头 | Dropout(0.2)，全连接 `512 → 11` | `11` 个 logits |

每个 BasicBlock 的主分支包含两次 3×3 卷积和 BatchNorm，第一次卷积后使用 ReLU。主分支与捷径分支相加后再经 ReLU，即 `y = ReLU(F(x) + S(x))`。尺寸相同时捷径为恒等映射；下采样或通道数变化时使用 stride 2 的 1×1 卷积和 BatchNorm 对齐。残差连接保留前层信息，并让梯度沿捷径传播。

当前 ResNet18 共 **11,182,155 个参数**，其中新分类头为 **5,643 个参数**（`512×11 + 11`）。该统计包含权重和 BatchNorm 的可学习参数，不包含 running mean/variance 等缓冲区。训练分两个阶段：

1. **分类头 warmup**：前 3 轮冻结全部主干，只更新分类头的 5,643 个参数，学习率为 1e-3。
2. **局部微调**：之后解冻 layer3、layer4 和分类头，共更新 10,499,083 个参数；主干初始学习率为 1e-4，分类头为 1e-3。stem、layer1、layer2 仍冻结。`--unfreeze all` 可改为微调全部主干。

默认微调保留预训练 BatchNorm 的运行统计量，避免小样本批次不断改写统计值；这不等同于冻结所有 BatchNorm 参数，已解冻层中的 affine 参数仍可更新。AdamW、余弦学习率调度和早停控制优化过程，具体参数见下表。

`--arch scratch` 对应初版 `NaiFamilyCNN`，当前 11 类时共 **614,771 个参数**。它包含四个 ConvBlock，输出通道为 24、48、96、160；每块为两次“3×3 卷积 → BatchNorm → ReLU”后接 2×2 最大池化，空间尺寸依次为 112、56、28、14。分类头为 `AdaptiveAvgPool(2×2) → Flatten(640) → Dropout → Linear(640,128) → ReLU → Dropout → Linear(128,11)`，训练默认 Dropout 为 0.4。

两个网络均输出原始 logits，不在模型内部附加 softmax：训练交叉熵直接接收 logits，测试和网页再用 softmax 计算匹配分数。网页的 **40% 拒识阈值**只影响结果展示，不改变网络参数、logits 或不带拒识的测试准确率。

### 4.2 基本命令

使用训练包中的固定数据：

```powershell
.\.venv\Scripts\python.exe train.py --device auto --output runs\local_001
```

使用自己的数据集：

```powershell
.\.venv\Scripts\python.exe train.py --data data_custom --device auto --output runs\custom_001
```

如需强制指定硬件，将 `auto` 改为 `cpu`、`cuda` 或 `xpu`。每个实验使用新的空输出目录。当前训练脚本不支持从 checkpoint 断点续训。

默认训练过程：前 3 轮只训练分类头；随后解冻 ResNet18 的 layer3/layer4，以较小的主干学习率微调，保留前层特征及 BatchNorm 统计量。增强仅用于训练集，验证和测试不使用随机增强。GPU 默认启用 FP16 AMP，CPU 使用 FP32。

### 4.3 常用超参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--arch` | `resnet18` | 网络结构，也可选择 `scratch` |
| `--epochs` | `40` | 最大训练轮数 |
| `--batch-size` | `16` | 每批图片数量 |
| `--lr` | `0.0001` | ResNet18 主干学习率；scratch 默认 0.001 |
| `--head-lr` | `0.001` | ResNet18 分类头学习率 |
| `--weight-decay` | `0.01` | AdamW 权重衰减 |
| `--dropout` | `0.2` | ResNet18 分类头 Dropout；scratch 默认 0.4 |
| `--warmup-epochs` | `3` | 预训练模型只训练分类头的轮数 |
| `--unfreeze` | `last2` | 微调后两组残差层；`all` 表示微调全部主干 |
| `--freeze-bn` | 启用 | 固定预训练 BatchNorm 统计量；`--no-freeze-bn` 关闭 |
| `--label-smoothing` | `0.05` | 训练交叉熵的标签平滑系数 |
| `--augmentation` | `gentle` | 轻量增强；也可选择初版的 `standard` |
| `--selection` | `accuracy` | 按验证准确率选权重，同分时按 loss；也可选择 `loss` |
| `--patience` | `12` | 验证选择指标连续不改善时的早停轮数 |
| `--workers` | `0` | 数据加载进程数，Windows 小数据集建议从 0 开始 |
| `--seed` | `42` | 训练随机种子 |
| `--no-amp` | 不启用此开关 | 关闭 GPU 混合精度，用于排查数值问题 |

显式指定参数的示例：

```powershell
.\.venv\Scripts\python.exe train.py --device xpu --epochs 40 --batch-size 16 --lr 0.0001 --head-lr 0.001 --dropout 0.2 --weight-decay 0.01 --label-smoothing 0.05 --warmup-epochs 3 --unfreeze last2 --patience 12 --seed 42 --output runs\tuned_001
```

建议每次只改变少量参数，并保持验证集不变。训练和验证均表现不足时，可逐步尝试微调更多层或调整学习率；训练准确率高而验证表现下降时，应检查重复样本、数据差异及正则化。显存不足时先降低 batch size。不要依据测试集反复选择参数或模型。

从零训练原始轻量 CNN：

```powershell
.\.venv\Scripts\python.exe train.py --arch scratch --lr 0.001 --dropout 0.4 --device auto --output runs\scratch_001
```

`--no-pretrained` 可让 ResNet18 从随机初始化开始训练，此时不使用预训练模型的分阶段冻结策略。小样本任务通常应先验证默认迁移学习方案。

### 4.4 训练输出

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 最优参数、结构、类别映射、预处理、归一化及数据划分 |
| `history.csv` | 各轮 train/val loss、准确率、学习率、阶段和耗时 |
| `loss_curve.png` | train loss 与 val loss 折线图 |
| `config.json` | 实验参数和实际使用的后端 |
| `manifest.json` | 本次训练的数据清单快照 |
| `summary.json` | 最佳轮次、完成轮数及训练耗时 |

训练交叉熵包含标签平滑，验证交叉熵不包含；因此不能仅按两条 loss 的绝对大小判断过拟合。权重只按验证指标选择，训练过程不使用测试准确率选模型。

## 5. 测试与单图推理

### 5.1 完整测试集评估

```powershell
.\.venv\Scripts\python.exe test.py --weights runs\local_001\best.pt --data data_processed --device auto --samples 12 --output test_results\local_001
```

准确率始终基于完整测试集，`--samples` 只控制随机展示的图片数量。输出 `report.json` 和 `predictions.png`，报告包含总体/逐类准确率、全部预测和混淆矩阵；矩阵行是真实类别，列是预测类别。

权重与测试数据必须使用相同类别映射、预处理及划分。若训练使用 `data_custom`，测试时也须指定 `--data data_custom`。

测试训练包内已保存的最优模型：

```powershell
.\.venv\Scripts\python.exe test.py --weights runs\resnet18_transfer_v1\best.pt --device auto --output test_results\verify_best
```

### 5.2 单张图片

```powershell
.\.venv\Scripts\python.exe test.py --weights runs\local_001\best.pt --image 'D:\images\example.png' --device auto
```

输出前三名类别及 softmax 分数，并写入 `test_results/prediction.json`。原始图片使用权重中保存的默认裁剪；可用 `--crop 0 0 0 0` 等四个比例覆盖默认裁剪。若输入已经处理为对应尺寸，添加 `--processed-image`，避免重复裁剪。单图真实标签未知时不计算准确率。

## 6. 本地运行网页与更新浏览器模型

仓库或训练包已包含网页所需全部静态资产。安装 Python 后即可预览：

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory docs
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。应通过 HTTP/HTTPS 访问，不能直接双击 `index.html`。首次加载约 43 MB 模型及约 12 MB 运行时。图片只在本机浏览器处理，不上传至服务器。

训练自己的权重后，安装导出依赖并重新生成网页模型：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe export_web.py --weights runs\local_001\best.pt --data data_processed
```

导出脚本验证 ONNX 与 PyTorch 在测试集上的数值和预测一致性，并生成模型 SHA-256 校验值。网页启动时核验模型文件完整性。展示图取自 `show/`，其中 `奶娃.png` 映射到“奶蛙”。页面对小奶图的右侧黑边作展示裁切，不改变原图或训练数据。

网页最高匹配度低于 40% 时显示“ops! 您上传的的图片中似乎没有奶家人”，并展示小奶；40% 及以上接受最高分候选。阈值位于 `docs/model-config.json` 的 `unknownThreshold`，导出脚本默认值也为 0.4。该分数未进行概率校准，阈值规则不等同于可靠的未知类别检测。浏览器 Canvas 与 Pillow 插值不同，分数可能有小幅差异。

## 7. GitHub Pages 部署

将代码和 `docs/` 推送到自己的 GitHub 仓库，在 Settings → Pages 中选择：

- Source：Deploy from a branch；
- Branch：`main`；
- Folder：`/docs`。

站点使用相对资源路径，可部署到仓库子路径。更新 `docs/` 后推送即可触发重新发布。若部署到自己的仓库，请同步修改网页和 README 中指向本项目的仓库链接。

`scripts/publish_github.py` 是本项目维护者的发布辅助工具，目标账户和仓库固定为 `bdlfy114514/naifamily-classifier`。其他用户不应直接使用该脚本发布自己的仓库；手动配置 Pages 即可。工具通过 Git Credential Manager 获取授权，不在项目中存放凭据。

## 8. 回归测试与训练包构建

```powershell
# 网页后处理和部署资产检查，需要 Node.js
node --test tests/web.test.mjs

# 网络兼容及分阶段训练检查
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 生成本地训练包，要求数据、最优权重和预训练缓存齐全
.\.venv\Scripts\python.exe scripts\package_training.py
```

打包输出为 `dist/naifamily-training-kit.zip`，并生成 `.zip.sha256` 校验文件。包内不包含 `.git/`、GitHub 登录信息、虚拟环境或临时文件。`dist/` 已加入 Git 忽略规则。

## 9. 实测结果与适用范围

固定数据划分为训练 163、验证 33、测试 33 张。默认 ResNet18 实验使用 XPU，在第 21 轮早停，第 9 轮的模型按验证集选为最优。

| 模型 | 完整测试集准确率 |
| --- | --- |
| 原始轻量 CNN | 26/33 = 78.79% |
| ResNet18 微调 | 32/33 = 96.97% |
| ONNX 导出模型 | 32/33 = 96.97%，与 PyTorch 预测一致 |

以上是未启用拒识时、当前固定划分的分类结果。网页启用阈值后会拒绝部分低分已知类别；不能将该结果视为任意上传图片的准确率。测试集较小，每张图片约影响 3.03 个百分点，不同硬件或数据划分的结果也可能变化。本项目用于学习和娱乐。

常见问题：

| 问题 | 处理方式 |
| --- | --- |
| `python` 指向 Windows Store，或找不到解释器 | 安装 Python 3.12，使用 `py -3.12` 或 `setup_env.ps1 -Python` |
| CUDA/XPU 不可用 | 检查 wheel 后端、硬件和驱动；可先用 `--device cpu` 验证流程 |
| 找不到 `manifest.json` | 解压完整训练包，或先对自己的原图执行预处理 |
| 输出目录非空 | 指定新的实验/预处理输出目录 |
| 数据与权重不匹配 | 使用训练时对应的数据目录，不重新随机划分后套用旧权重 |
| loss 出现非有限值 | 尝试 `--no-amp` 并降低学习率 |
| 网页模型无法加载 | 使用 HTTP/HTTPS，确认模型、配置及 `vendor/` 文件完整 |

依赖与参考：[PyTorch 安装](https://pytorch.org/get-started/previous-versions/)、[torchvision ResNet](https://docs.pytorch.org/vision/stable/models/resnet.html)、[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)。浏览器运行时许可证和第三方声明保存在 `docs/vendor/`。
