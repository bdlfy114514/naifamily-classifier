"""裁边、缩放、去重并按类别划分数据；绝不修改源图。只依赖 Pillow。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
DEFAULT_CONFIG = {
    'size': 224,
    'background': [255, 255, 255],
    'default_crop': [0.0, 0.0, 0.0, 0.06],  # left, top, right, bottom
    'class_crops': {},
    'file_crops': {},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_config(config):
    if not isinstance(config['size'], int) or config['size'] < 32:
        raise ValueError('size 必须是 >=32 的整数')
    bg = config['background']
    if len(bg) != 3 or any(not isinstance(x, int) or not 0 <= x <= 255 for x in bg):
        raise ValueError('background 必须是三个 0~255 的整数')
    for crop in [config['default_crop'], *config['class_crops'].values(), *config['file_crops'].values()]:
        if len(crop) != 4 or any(not isinstance(x, (int, float)) or not 0 <= x < 1 for x in crop):
            raise ValueError('裁剪须为 [左, 上, 右, 下] 四个 0~1 比例')
        if crop[0] + crop[2] >= .8 or crop[1] + crop[3] >= .8:
            raise ValueError('裁剪过大：至少保留原图宽、高的 20%')


def load_config(path=None, size=None):
    config = {**DEFAULT_CONFIG, **(read_json(path) if path else {})}
    if size is not None:
        config['size'] = size
    validate_config(config)
    return config


def prepare_image(path, config, relative_name=None):
    """原图推理与离线预处理共用此函数。推理未知类别时只使用默认裁剪。"""
    validate_config(config)
    name = relative_name.replace('\\', '/') if relative_name else ''
    class_name = name.split('/')[0] if '/' in name else ''
    crop = config['file_crops'].get(name, config['class_crops'].get(class_name, config['default_crop']))
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', image.size, (*config['background'], 255))
        image = Image.alpha_composite(background, image).convert('RGB')
    w, h = image.size
    left, top, right, bottom = crop
    image = image.crop((round(w * left), round(h * top), round(w * (1-right)), round(h * (1-bottom))))
    return ImageOps.pad(image, (config['size'], config['size']), method=Image.Resampling.LANCZOS,
                        color=tuple(config['background']), centering=(.5, .5))


def image_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def find_classes(root, excluded):
    folders = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.') or folder.resolve() in excluded:
            continue
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        # 只有直接包含图片的一级目录作为类别；排除 outputs/artifacts 等生成目录。
        if any(p.parent == folder for p in files):
            folders.append((folder, files))
    if len(folders) < 2:
        raise ValueError('至少需要两个含图片的类别子文件夹')
    return folders


def save_preview(pairs, target, size):
    cols = 4
    cell_w, cell_h = size * 2 + 12, size + 32
    canvas = Image.new('RGB', (cols * cell_w, ((len(pairs)+cols-1)//cols) * cell_h), '#e5e7eb')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (path, processed, title) in enumerate(pairs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        with Image.open(path) as raw:
            raw = ImageOps.pad(ImageOps.exif_transpose(raw).convert('RGB'), (size, size), color='white')
        canvas.paste(raw, (x, y))
        canvas.paste(processed.resize((size, size), Image.Resampling.LANCZOS), (x + size + 6, y))
        draw.text((x+3, y+size+3), f'{title} | original / processed', font=font, fill='black')
    canvas.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data_processed')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'preprocess_config.json')
    parser.add_argument('--size', type=int, help='覆盖配置中的输出大小')
    parser.add_argument('--val-ratio', type=float, default=.15)
    parser.add_argument('--test-ratio', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preview-only', action='store_true', help='只生成每类前三张的前后对照，不划分数据')
    args = parser.parse_args()
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        parser.error('验证/测试比例须大于 0 且和小于 1')
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source.is_relative_to(output):
        parser.error('输出目录不能是源目录或源目录的父目录')
    if output.exists() and any(output.iterdir()):
        parser.error('输出目录非空；请指定新的 --output，避免混入旧数据')
    config = load_config(args.config, args.size)
    folders = find_classes(source, {output, (source/'artifacts').resolve(), (source/'runs').resolve()})
    classes = [folder.name for folder, _ in folders]
    output.mkdir(parents=True, exist_ok=True)
    preview = []
    if args.preview_only:
        for folder, files in folders:
            for i, path in enumerate(files[:3]):
                preview.append((path, prepare_image(path, config, path.relative_to(source).as_posix()), f'{folder.name} #{i+1}'))
        save_preview(preview, output/'preview.jpg', min(config['size'], 160))
        # preview 的输入尺寸保持一致。
        write_json(output/'preprocess_config.json', config)
        print(f'预览已保存到 {output / "preview.jpg"}')
        return

    rng = random.Random(args.seed)
    records, duplicates, seen = [], [], {}
    # 在写入划分之前检查全部图片；坏图或跨类别重复时明确终止。
    grouped = defaultdict(list)
    for folder, files in folders:
        for i, path in enumerate(files):
            rel = path.relative_to(source).as_posix()
            try:
                image = prepare_image(path, config, rel)
            except Exception as error:
                raise RuntimeError(f'无法处理图片 {rel}: {error}') from error
            digest = image_digest(image)
            if digest in seen:
                old_rel, old_class = seen[digest]
                if old_class != folder.name:
                    raise ValueError(f'相同图片被标为不同类别：{old_rel} / {rel}')
                duplicates.append({'source': rel, 'same_as': old_rel})
                continue
            seen[digest] = (rel, folder.name)
            grouped[folder.name].append((path, rel, digest))
            if i < 2:
                preview.append((path, image, f'{folder.name} #{i+1}'))
        if len(grouped[folder.name]) < 5:
            raise ValueError(f'{folder.name} 去重后少于 5 张，不能可靠地划分三个集合')

    for class_name in classes:
        items = grouped[class_name]
        rng.shuffle(items)
        n = len(items)
        n_val, n_test = max(1, round(n*args.val_ratio)), max(1, round(n*args.test_ratio))
        if n_val+n_test >= n:
            raise ValueError(f'{class_name} 的划分比例未留出训练样本')
        splits = ['val']*n_val + ['test']*n_test + ['train']*(n-n_val-n_test)
        for (path, rel, digest), split in zip(items, splits):
            dest = Path(split)/class_name/(digest[:20]+'.png')
            (output/dest).parent.mkdir(parents=True, exist_ok=True)
            prepare_image(path, config, rel).save(output/dest)
            records.append({'source': rel, 'path': dest.as_posix(), 'class': class_name,
                            'split': split, 'sha256': digest})
    manifest = {'schema_version': 1, 'classes': classes, 'class_to_idx': {c:i for i,c in enumerate(classes)},
                'preprocess': config, 'seed': args.seed, 'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio, 'records': records, 'duplicates_removed': duplicates}
    write_json(output/'manifest.json', manifest)
    write_json(output/'preprocess_config.json', config)
    save_preview(preview, output/'preview.jpg', config['size'])
    counts = Counter(r['split'] for r in records)
    print(f'类别数: {len(classes)} | 去重后: {len(records)} | 删除重复副本: {len(duplicates)}')
    print(f'划分: {dict(counts)}\n输出: {output}\n请检查 preview.jpg，必要时修改裁剪配置后输出到新目录。')


if __name__ == '__main__':
    main()
