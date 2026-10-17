# 奶家人识别器

上传一张图片，识别它是哪位奶家人，并显示 `show/` 提供的对应展示图。网页在浏览器本地运行最优 ResNet18 模型，**图片不上传到服务器**。可直接部署到 GitHub Pages，无需 Python 后端和付费推理接口。

**[打开奶家人识别器 →](https://bdlfy114514.github.io/naifamily-classifier/)**

仓库：[`bdlfy114514/naifamily-classifier`](https://github.com/bdlfy114514/naifamily-classifier)。网页标题为“奶家人识别器”。

## 网页功能

- 上传或拖入 JPG、PNG、WebP 等图片，支持预览、换图和重复识别。
- 显示预测类别、对应展示图，以及前三名的匹配度。
- 提供 11 位奶家人的图鉴，点击可直接试用识别。
- 最高匹配度低于 **55%** 时显示：**ops! 您上传的的图片中似乎没有奶家人**，并展示小奶图。
- `show/奶娃.png` 映射到模型的“奶蛙”类别，其余同名映射。
- 支持窄屏布局、模型下载进度、加载失败重试、文件格式和大小检查。

匹配度是 softmax 分数，不是经过校准的真实概率。55% 是可修改的拒识阈值，不保证过滤所有陌生图片；该模型只有 11 个已知类别，未进行专门的未知类训练。阈值配置位于 `docs/model-config.json` 的 `unknownThreshold` 字段。

## 本地预览

在项目目录运行：

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory docs
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。不能直接双击 `index.html`，模型需要通过 HTTP/HTTPS 加载。

首次加载约 43 MB 模型及约 12 MB 推理运行时；在加载完成后可连续识别。所有运行时文件均打包在站点中，不依赖外部 CDN。现代浏览器需要支持 WebAssembly 和 `createImageBitmap`。

## GitHub Pages 发布

站点目录为 `docs/`，全部资源采用相对路径，支持仓库子路径。
GitHub 仓库名不能使用中文，实际仓库使用用户指定的 `naifamily-classifier`。

在 Git Credential Manager 官方流程登录：

```powershell
git credential-manager github login --device --no-ui --username bdlfy114514
```

随后运行：

```powershell
.\.venv\Scripts\python.exe scripts\publish_github.py
```

脚本会核验账户、运行网页测试、创建或使用目标公开仓库、提交选定源文件并推送，最后将 Pages 设置为 `main` 分支的 `/docs` 目录。不会读取或打印密码，不在项目文件中保存 GitHub token。已有不同的 origin 或远端提交冲突时会停止，不进行强制推送。

也可手动推送代码后，在仓库 Settings → Pages → Deploy from a branch 选择 `main` 和 `/docs`。GitHub 会自动执行 Pages 构建发布。

预期网址为 `https://bdlfy114514.github.io/naifamily-classifier/`；以实际发布状态为准。

## 模型与验证

保持原有 train/val/test＝163/33/33 划分，最优模型来自 ResNet18 微调第 9 轮；训练于第 21 轮早停。

| 模型 | 原测试集分类准确率 |
| --- | --- |
| 初版轻量 CNN | 26/33＝78.79% |
| 最优 ResNet18 | 32/33＝96.97% |
| 导出的 ONNX | 32/33＝96.97%，与 PyTorch 预测一致 |

该准确率针对不带拒识的完整测试集分类。网页启用 55% 拒识规则后，会拒绝部分低分的已知类别图片，因此不能把 96.97% 直接理解为网页对任意上传图片的准确率。验证集 33 张中有 32 张超过当前阈值。浏览器 Canvas 与 Pillow 的缩放插值不同，也可能导致分数存在小幅差异。

最优 Python 权重保存在本机 `runs/resnet18_transfer_v1/best.pt`；训练日志和 loss 图在同目录，测试报告位于 `test_results/resnet18_transfer_v1/`。这些本机训练产物不上传到公开仓库；网页使用的模型为随站点提交的 `docs/assets/naifamily.onnx`。

## 项目目录

```text
docs/                      GitHub Pages 网站、展示图、ONNX 模型及推理运行时
show/                      用户提供的原版展示图
scripts/                   网页运行时获取与 GitHub 发布工具
tests/                     Python 网络与 JavaScript 后处理回归测试
model.py / train.py        CNN 定义及两阶段微调
common.py / test.py         数据读取、完整测试集评估、单图推理
preprocess.py              外部原始图片集的预处理脚本
export_web.py              导出模型、生成网页展示图及数值一致性检查
requirements-web.txt       模型导出额外依赖
setup_env.ps1              CPU / CUDA / XPU 虚拟环境配置
```

本机额外保留 `.venv/`、`data_processed/`、唯一最优 `runs/`、对应 `test_results/` 和少量 `artifacts/` 检查记录。11 个原始训练图目录、旧训练记录及旧测试输出已按要求删除。预处理后的数据和虚拟环境不上传。

## 重新训练与导出

默认参数：预训练 ResNet18；分类头 warmup 3 轮，然后微调 layer3/layer4；主干学习率 1e-4、分类头 1e-3；冻结 BatchNorm 统计量；Dropout 0.2、label smoothing 0.05、weight decay 0.01；轻量增强、余弦学习率调度。按验证准确率选权重，同分取验证 loss 更低者，不按测试集选模型。

```powershell
.\.venv\Scripts\python.exe train.py --device xpu --output runs\new_experiment
.\.venv\Scripts\python.exe test.py --weights runs\new_experiment\best.pt --device xpu
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe export_web.py --weights runs\new_experiment\best.pt
```

`--device cuda` 支持 NVIDIA GPU；需相应版本 PyTorch。旧网络和旧权重仍可通过 `--arch scratch` 及自动结构读取兼容。原始数据已删除，如需重新预处理新图片，请使用 `preprocess.py --source 外部数据目录 --output 新目录`，不要对 `show/` 展示图进行数据集划分。

运行时更新/重新获取：

```powershell
.\.venv\Scripts\python.exe scripts\fetch_web_runtime.py
```

模型导出会检查 ONNX 格式与全部测试图预测一致性，并生成 SHA-256 校验值。网页启动时也会检查模型版本和完整性。

## 测试

```powershell
node --test tests/web.test.mjs
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

JavaScript 检查稳定 softmax、高匹配类别及图像映射、低匹配小奶回退、无效输出和所有部署资产的完整性。Python 检查旧模型兼容、推理不联网、分阶段训练及 BatchNorm 冻结。

前端推理基于 [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)；运行时版本固定为 1.23.2，MIT 许可证和第三方声明见 `docs/vendor/`。预训练 CNN 使用 [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/resnet.html)。本站为娱乐项目。
