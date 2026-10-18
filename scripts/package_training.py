"""生成可迁移的本地训练包：源代码、处理后数据、最优权重及预训练初始化权重。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'naifamily-training-kit'
ROOT_FILES = ['README.md', '.gitignore', 'package.json', 'model.py', 'common.py', 'train.py',
              'test.py', 'preprocess.py', 'preprocess_config.json', 'export_web.py',
              'setup_env.ps1', 'requirements.txt', 'requirements-web.txt', 'requirements-lock.txt']
DIRECTORIES = ['scripts', 'tests', 'reports', 'docs', 'show', 'data_processed',
               'runs/resnet18_transfer_v1', 'test_results/resnet18_transfer_v1',
               'runs/resnet18_face_oe_v1', 'hard_negatives/lfw']
PRETRAINED = '.cache/torch/checkpoints/resnet18-f37072fd.pth'


def collect_files():
    required = [*ROOT_FILES, 'data_processed/manifest.json',
                'runs/resnet18_transfer_v1/best.pt', 'runs/resnet18_face_oe_v1/best.pt',
                'hard_negatives/lfw/manifest.json', PRETRAINED]
    missing = [name for name in required if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError(f'训练包缺少必要文件：{missing}')
    manifest = json.loads((ROOT/'data_processed/manifest.json').read_text(encoding='utf8'))
    for record in manifest['records']:
        if not (ROOT/'data_processed'/record['path']).is_file():
            raise FileNotFoundError(f'数据图片缺失：{record["path"]}')
    files = {ROOT/name for name in ROOT_FILES}
    files.add(ROOT/PRETRAINED)
    for name in DIRECTORIES:
        folder = ROOT/name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files.update(p for p in folder.rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'拒绝打包越界路径或符号链接：{path}')
    return sorted(files, key=lambda p:p.relative_to(ROOT).as_posix()), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/f'{PREFIX}.zip')
    args = parser.parse_args()
    files, data = collect_files()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.zip.tmp')
    entries = []
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            name = file.relative_to(ROOT).as_posix()
            payload = file.read_bytes()
            entries.append({'path':name, 'bytes':len(payload), 'sha256':hashlib.sha256(payload).hexdigest()})
            archive.writestr(f'{PREFIX}/{name}', payload)
        package_manifest = {'schemaVersion':1, 'createdUtc':datetime.now(timezone.utc).isoformat(),
            'classes':data['classes'], 'splits':{s:sum(r['split']==s for r in data['records']) for s in ('train','val','test')},
            'containsProcessedData':True, 'containsBestWeights':True, 'containsPretrainedInitialization':True,
            'containsPythonRuntimeOrVenv':False, 'dependencyInstallationRequiresNetwork':True,
            'files':entries}
        archive.writestr(f'{PREFIX}/KIT_MANIFEST.json', json.dumps(package_manifest, ensure_ascii=False, indent=2).encode('utf8'))
    with zipfile.ZipFile(temporary) as archive:
        damaged = archive.testzip()
        if damaged:
            raise RuntimeError(f'压缩包 CRC 校验失败：{damaged}')
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(f'{digest}  {args.output.name}\n', encoding='ascii')
    print(json.dumps({'archive':str(args.output.resolve()), 'bytes':args.output.stat().st_size,
                      'files':len(entries)+1, 'sha256':digest, 'splits':package_manifest['splits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
