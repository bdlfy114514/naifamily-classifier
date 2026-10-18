"""Build separate script-only training kits and a local-only dataset archive."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREFIX='naifamily-training-kit'
CORE=['LICENSE','model.py','common.py','train.py','test.py','preprocess.py','preprocess_config.json',
      'setup_env.ps1','requirements.txt','requirements-lock.txt']
EXTRAS=['README.md','TRAINING_QUICKSTART.md','scripts/train_face_rejection.py',
        'scripts/prepare_face_negatives.py','scripts/prepare_cartoon_negatives.py']
DATA_ROOTS=['data_processed_v12','hard_negatives/mixed_v12']

def checked_file(path):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise ValueError(f'Path outside workspace: {path}')
    if not path.is_file():raise FileNotFoundError(path)
    return path

def script_payloads(ref=None):
    if ref:
        payloads={}
        for name in CORE:
            result=subprocess.run(['git','show',f'{ref}:{name}'],cwd=ROOT,capture_output=True)
            if result.returncode:
                if name=='LICENSE':payloads[name]=checked_file(ROOT/name).read_bytes();continue
                raise RuntimeError(f'Missing historical script: {ref}:{name}')
            payloads[name]=result.stdout
        guide=f'''# {ref} 历史版训练脚本包

本包只含 {ref} 的必要训练、预处理、测试脚本及依赖配置，不含数据集、模型权重、网页、日志或虚拟环境。请在单独目录解压，避免覆盖其他版本。

数据集由维护者在其他位置提供。若使用当前 naifamily-dataset-v1.2.zip，将它解压到同一个父目录，合并 naifamily-training-kit/，得到 data_processed_v12/。当前数据不是最初版本的数据快照，不能用于声称复现历史分数。

在 naifamily-training-kit/ 内执行：

```powershell
.\\setup_env.ps1 -Backend xpu
.\\.venv\\Scripts\\python.exe train.py --data data_processed_v12 --output runs\\local_base --device auto
.\\.venv\\Scripts\\python.exe test.py --data data_processed_v12 --weights runs\\local_base\\best.pt --output test_results\\local_base
```

CUDA 用 -Backend cuda，CPU 用 -Backend cpu。首次安装依赖和预训练初始化权重需要联网。本历史脚本只进行已知类别分类训练；当前负样本训练功能请使用最新版本的脚本包。
'''
        payloads['README.md']=guide.encode('utf8')
        return payloads
    return {name:checked_file(ROOT/name).read_bytes() for name in CORE+EXTRAS}

def dataset_payloads():
    payloads={};summaries={}
    for directory in DATA_ROOTS:
        root=ROOT/directory
        manifest=json.loads(checked_file(root/'manifest.json').read_text(encoding='utf8'))
        for record in manifest['records']:checked_file(root/record['path'])
        summaries[directory]=dict(Counter(r['split'] for r in manifest['records']))
        for file in root.rglob('*'):
            if file.is_file():payloads[file.relative_to(ROOT).as_posix()]=checked_file(file).read_bytes()
    return payloads,summaries

def build(output,payloads,metadata,manifest_name):
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.zip.tmp');entries=[]
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payloads.items()):
            entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            z.writestr(f'{PREFIX}/{name}',data)
        manifest=dict(schemaVersion=2,createdUtc=datetime.now(timezone.utc).isoformat(),files=entries,**metadata)
        z.writestr(f'{PREFIX}/{manifest_name}',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf8'))
    with zipfile.ZipFile(temp) as z:
        if damaged:=z.testzip():raise RuntimeError(f'ZIP CRC failed: {damaged}')
    temp.replace(output);digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n',encoding='ascii')
    print(json.dumps(dict(archive=str(output.resolve()),bytes=output.stat().st_size,files=len(entries)+1,sha256=digest,**metadata),ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['scripts','dataset','all'],default='all')
    p.add_argument('--ref',help='Build script-only historical kit from an existing Git ref')
    p.add_argument('--output',type=Path,help='Only with one --kind')
    a=p.parse_args()
    if a.output and a.kind=='all':p.error('--output requires --kind scripts or dataset')
    if a.ref and a.kind!='scripts':p.error('--ref requires --kind scripts')
    if a.kind in ('scripts','all'):
        payloads=script_payloads(a.ref)
        if any(Path(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.pt','.pth','.onnx'} for n in payloads):raise ValueError('Non-script asset in training kit')
        output=a.output or ROOT/'dist'/('naifamily-training-kit-'+a.ref+'.zip' if a.ref else 'naifamily-training-kit.zip')
        build(output,payloads,dict(packageType='training-scripts',sourceRef=a.ref or 'working-tree',containsProcessedData=False,
              containsBestWeights=False,containsPretrainedInitialization=False,containsPythonRuntimeOrVenv=False,
              dependencyInstallationRequiresNetwork=True),'KIT_MANIFEST.json')
    if a.kind in ('dataset','all'):
        payloads,splits=dataset_payloads()
        build(a.output or ROOT/'dist/naifamily-dataset-v1.2.zip',payloads,
              dict(packageType='dataset',containsProcessedData=True,containsBestWeights=False,containsPretrainedInitialization=False,
                   splits=splits,publishDestination='User-managed external storage; do not upload to GitHub Release'),'DATASET_MANIFEST.json')

if __name__=='__main__':main()
