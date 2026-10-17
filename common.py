"""训练与测试共享的数据读取、设备选择和归一化。"""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from preprocess import image_digest, read_json

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def select_device(requested='auto'):
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
    if requested == 'auto':
        requested = 'cuda' if cuda else 'xpu' if xpu else 'cpu'
    if requested == 'cuda' and not cuda:
        raise RuntimeError('CUDA 不可用：请安装 CUDA 版 PyTorch，并检查 NVIDIA 显卡驱动。')
    if requested == 'xpu' and not xpu:
        raise RuntimeError('XPU 不可用：请安装 XPU 版 PyTorch，并检查 Intel GPU 驱动。')
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    # 固定随机序列；不同后端浮点结果可能有微小差别。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_transform(training=False, mean=MEAN, std=STD, augmentation='standard'):
    steps = []
    if training:
        gentle = augmentation == 'gentle'
        steps = [transforms.RandomHorizontalFlip(),
                 transforms.RandomAffine(degrees=7 if gentle else 10,
                                         translate=(.03, .03) if gentle else (.05, .05),
                                         scale=(.95, 1.05) if gentle else (.9, 1.1),
                                         interpolation=transforms.InterpolationMode.BILINEAR, fill=255),
                 transforms.ColorJitter(brightness=.1 if gentle else .15,
                                        contrast=.1 if gentle else .15,
                                        saturation=.08 if gentle else .12, hue=.01 if gentle else .015)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def load_manifest(root):
    root = Path(root).resolve()
    path = root/'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'找不到 {path}；请先执行 preprocess.py')
    manifest = read_json(path)
    classes = manifest['classes']
    if manifest['class_to_idx'] != {c:i for i,c in enumerate(classes)}:
        raise ValueError('manifest 类别映射不一致')
    seen_paths, seen_hashes = set(), set()
    counts = {(split, c): 0 for split in ('train', 'val', 'test') for c in classes}
    size = manifest['preprocess']['size']
    for record in manifest['records']:
        key = (record['split'], record['class'])
        if key not in counts:
            raise ValueError(f'非法类别或划分：{record}')
        file = (root/record['path']).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f'图片路径越界或不存在：{record["path"]}')
        if record['path'] in seen_paths or record['sha256'] in seen_hashes:
            raise ValueError('manifest 存在重复图片，可能导致数据泄漏')
        with Image.open(file) as image:
            if image.mode != 'RGB' or image.size != (size, size) or image_digest(image) != record['sha256']:
                raise ValueError(f'预处理图片被修改或与 manifest 不符：{file}')
        counts[key] += 1
        seen_paths.add(record['path'])
        seen_hashes.add(record['sha256'])
    if any(n == 0 for n in counts.values()):
        raise ValueError('每个类别在 train/val/test 中都必须至少有一张图片')
    return manifest


class ManifestDataset(Dataset):
    def __init__(self, root, manifest, split, transform):
        self.root = Path(root)
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.mapping = manifest['class_to_idx']
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root/record['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.mapping[record['class']]
