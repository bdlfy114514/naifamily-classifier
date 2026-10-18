"""读取 best.pt，计算独立测试集准确率并展示随机样本；支持单张原图分类。"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from common import ManifestDataset, build_transform, load_manifest, select_device
from model import build_model
from preprocess import prepare_image, validate_config, write_json


def set_plot_font():
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False


def predict(model, batch, device):
    with torch.inference_mode():
        return model(batch.to(device)).softmax(dim=1).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path(__file__).parent/'data_processed_v12')
    p.add_argument('--device', choices=['auto', 'cuda', 'xpu', 'cpu'], default='auto')
    p.add_argument('--samples', type=int, default=12, help='随机展示数量；准确率始终基于完整测试集')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'test_results')
    p.add_argument('--image', type=Path, help='分类任意一张原始图片；此模式不计算准确率')
    p.add_argument('--processed-image', action='store_true', help='--image 已经按相同配置预处理，跳过裁边')
    p.add_argument('--crop', nargs=4, type=float, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                   help='单张原图裁边比例；覆盖默认裁剪')
    args = p.parse_args()
    if args.samples < 0 or args.batch_size < 1:
        p.error('samples 须非负，batch-size 须 >=1')
    if args.processed_image and args.image is None:
        p.error('--processed-image 必须与 --image 同用')
    if args.crop is not None and (args.image is None or args.processed_image):
        p.error('--crop 只能用于 --image 原图模式')
    if not args.weights.is_file():
        p.error(f'权重不存在：{args.weights}，请先训练生成 best.pt')
    device = select_device(args.device)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    model = build_model(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model = model.to(device).eval()
    normalization = checkpoint['normalization']
    transform = build_transform(mean=normalization['mean'], std=normalization['std'])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image:
        if args.processed_image:
            from PIL import Image
            with Image.open(args.image) as source:
                image = source.convert('RGB')
            size = checkpoint['preprocess']['size']
            if image.size != (size, size):
                p.error(f'预处理图像尺寸须为 {size}×{size}')
        else:
            config = {**checkpoint['preprocess']}
            if args.crop is not None:
                config['default_crop'] = args.crop
                validate_config(config)
            image = prepare_image(args.image, config)
        probabilities = predict(model, transform(image).unsqueeze(0), device)[0]
        top = torch.topk(probabilities, min(3, len(classes)))
        result = {'image': str(args.image.resolve()), 'predicted_class': classes[top.indices[0].item()],
                  'top3': [{'class': classes[i], 'probability': score} for score,i in zip(top.values.tolist(), top.indices.tolist())]}
        write_json(args.output/'prediction.json', result)
        for item in result['top3']:
            print(f'{item["class"]}: {item["probability"]:.2%}')
        return

    manifest = load_manifest(args.data)
    if (manifest['classes'] != classes or manifest['class_to_idx'] != checkpoint['class_to_idx']
            or manifest['preprocess'] != checkpoint['preprocess']):
        raise ValueError('数据集类别映射或预处理配置与权重不一致')
    # 比较全部划分，防止重新划分后的旧训练图片被当成测试图片。
    if manifest['records'] != checkpoint['manifest']['records']:
        raise ValueError('数据划分与训练时不同，请使用生成该权重时的数据集')
    dataset = ManifestDataset(args.data, manifest, 'test', transform)
    predictions, labels, confidence = [], [], []
    for batch, target in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        scores = predict(model, batch, device)
        values, indices = scores.max(1)
        predictions.extend(indices.tolist())
        labels.extend(target.tolist())
        confidence.extend(values.tolist())
    matrix = [[0]*len(classes) for _ in classes]
    rows = []
    for record, true, pred, conf in zip(dataset.records, labels, predictions, confidence):
        matrix[true][pred] += 1
        rows.append({'path': record['path'], 'true_class': classes[true], 'predicted_class': classes[pred],
                     'confidence': conf, 'correct': true == pred})
    correct = sum(x == y for x,y in zip(labels, predictions))
    accuracy = correct/len(labels)
    selected = random.Random(args.seed).sample(range(len(dataset)), min(args.samples, len(dataset)))
    report = {'accuracy': accuracy, 'correct': correct, 'total': len(dataset), 'classes': classes,
              'confusion_matrix': matrix, 'confusion_matrix_axes': 'rows=true, columns=predicted',
              'per_class_accuracy': {c: matrix[i][i]/sum(matrix[i]) for i,c in enumerate(classes)},
              'predictions': rows, 'sample_indices': selected, 'seed': args.seed,
              'weights': str(args.weights.resolve()), 'best_epoch': checkpoint['epoch']}
    write_json(args.output/'report.json', report)
    print(f'Test accuracy: {correct}/{len(labels)} = {accuracy:.2%} | device={device}')
    for index in selected:
        row = rows[index]
        print(f'{row["path"]} | true={row["true_class"]} | pred={row["predicted_class"]} '
              f'({row["confidence"]:.2%}) | {"OK" if row["correct"] else "WRONG"}')
    if selected:
        from PIL import Image
        set_plot_font()
        cols = min(4, len(selected))
        fig, axes = plt.subplots((len(selected)+cols-1)//cols, cols, figsize=(cols*3.4, ((len(selected)+cols-1)//cols)*3.8), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, index in zip(axes.flat, selected):
            row = rows[index]
            with Image.open(args.data/row['path']) as image:
                ax.imshow(image.copy())
            ax.set_title(f'True: {row["true_class"]}\nPred: {row["predicted_class"]} ({row["confidence"]:.1%})',
                         color='green' if row['correct'] else 'red')
        fig.suptitle(f'Test accuracy: {correct}/{len(labels)} = {accuracy:.2%}')
        fig.tight_layout()
        fig.savefig(args.output/'predictions.png', dpi=150)
        plt.close(fig)
    print(f'Report: {args.output/"report.json"}')


if __name__ == '__main__':
    main()
