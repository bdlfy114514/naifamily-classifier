"""从 npm 官方仓库获取固定版本 ONNX Runtime Web，页面无需依赖第三方 CDN。"""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request

VERSION = '1.23.2'
ROOT = Path(__file__).resolve().parents[1]
FILES = ['ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']


def main():
    url = f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz'
    request = urllib.request.Request(url, headers={'User-Agent': 'naifamily-classifier-build'})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    output = ROOT/'docs'/'vendor'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for name in FILES:
            data = archive.extractfile(f'package/dist/{name}').read()
            (output/name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        license_names = [m.name for m in archive.getmembers() if m.name in ('package/LICENSE', 'package/LICENSE.txt')]
        if license_names:
            (output/'LICENSE.txt').write_bytes(archive.extractfile(license_names[0]).read())
    if not license_names:
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/LICENSE', timeout=30) as response:
            (output/'LICENSE.txt').write_bytes(response.read())
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/ThirdPartyNotices.txt', timeout=30) as response:
        (output/'ThirdPartyNotices.txt').write_bytes(response.read())
    (output/'version.json').write_text(json.dumps({'version': VERSION, 'source': url, 'sha256': hashes}, indent=2), encoding='utf8')
    print(json.dumps({'version': VERSION, 'files': FILES, 'totalBytes': sum((output/n).stat().st_size for n in FILES)}))


if __name__ == '__main__':
    main()
