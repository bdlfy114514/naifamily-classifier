"""导出最优模型和展示图，供纯静态 GitHub Pages 网页使用。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps
import torch

from common import ManifestDataset, build_transform, load_manifest
from model import build_model
from preprocess import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('runs/resnet18_face_oe_v1/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('docs'))
    parser.add_argument('--data', type=Path, default=Path('data_processed'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    model = build_model(**checkpoint['model_config']).eval()
    model.load_state_dict(checkpoint['model_state'])
    assets = args.output/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    classes = checkpoint['classes']
    class_assets = []
    for index, name in enumerate(classes):
        # 用户提供的 show/奶娃.png 对应模型中的奶蛙。
        source = Path('show')/f'{name}.png'
        if not source.exists() and name == '奶蛙':
            source = Path('show/奶娃.png')
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert('RGBA')
            background = Image.new('RGBA', image.size, 'white')
            image = Image.alpha_composite(background, image).convert('RGB')
            image.thumbnail((720, 720), Image.Resampling.LANCZOS)
            dest = assets/f'family-{index:02d}.webp'
            image.save(dest, 'WEBP', quality=90, method=6)
        class_assets.append({'name': name, 'image': f'assets/{dest.name}'})
    size = checkpoint['preprocess']['size']
    output = assets/'naifamily.onnx'
    # 稳定的 legacy 导出路径；仅普通 ONNX 算子，无 Python/自定义依赖。
    torch.onnx.export(model, torch.zeros(1,3,size,size), str(output), opset_version=17,
                      input_names=['images'], output_names=['logits'],
                      dynamic_axes={'images': {0:'batch'}, 'logits': {0:'batch'}}, dynamo=False)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    manifest = load_manifest(args.data)
    if manifest != checkpoint['manifest']:
        raise ValueError('导出验证所用数据集与最优权重的划分不一致')
    dataset = ManifestDataset(args.data, manifest, 'test', build_transform(
        mean=checkpoint['normalization']['mean'], std=checkpoint['normalization']['std']))
    images, labels = zip(*(dataset[i] for i in range(len(dataset))))
    batch = torch.stack(images)
    with torch.inference_mode():
        torch_logits = model(batch).numpy()
    onnx_logits = session.run(['logits'], {'images': batch.numpy()})[0]
    np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(torch_logits.argmax(1), onnx_logits.argmax(1))
    correct = int((onnx_logits.argmax(1) == np.array(labels)).sum())
    # 用验证集了解分数比规则的接受比例，不以测试标签调整规则。
    val = ManifestDataset(args.data, manifest, 'val', build_transform())
    val_batch = torch.stack([val[i][0] for i in range(len(val))])
    val_logits = session.run(['logits'], {'images': val_batch.numpy()})[0]
    val_prob = np.exp(val_logits-val_logits.max(1,keepdims=True))
    val_prob /= val_prob.sum(1,keepdims=True)
    min_top_score_ratio = 1.5
    metadata = {
        'schemaVersion': 1, 'classes': class_assets, 'inputSize': size,
        'mean': list(checkpoint['normalization']['mean']), 'std': list(checkpoint['normalization']['std']),
        'crop': checkpoint['preprocess']['default_crop'], 'background': checkpoint['preprocess']['background'],
        'model': 'assets/naifamily.onnx', 'modelBytes': output.stat().st_size,
        'modelSha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'minTopScoreRatio': min_top_score_ratio, 'fallbackClass': '小奶',
        'unknownMessage': 'ops! 您上传的的图片中似乎没有奶家人',
        'metrics': {'correct': correct, 'total': len(labels), 'accuracy': correct/len(labels)},
        'source': {'architecture': checkpoint['model_config']['arch'], 'bestEpoch': checkpoint['epoch'],
                   'trainingMethod': checkpoint.get('method', 'supervised_transfer')},
    }
    write_json(args.output/'model-config.json', metadata)
    report = {
        'maxAbsLogitError': float(np.max(np.abs(torch_logits-onnx_logits))),
        'samePredictions': True, 'onnxTestCorrect': correct, 'testTotal': len(labels),
        'minTopScoreRatio': min_top_score_ratio,
        'validationAccepted': int((np.sort(val_prob, axis=1)[:, -1] >
                                   np.sort(val_prob, axis=1)[:, -2] * min_top_score_ratio).sum()),
        'validationTotal': len(val), 'modelSha256': metadata['modelSha256'],
        'note': '分数比规则不等同于可靠未知检测；人脸负样本测试不覆盖所有未知输入。',
    }
    write_json(Path('artifacts/web_export_report.json'), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
