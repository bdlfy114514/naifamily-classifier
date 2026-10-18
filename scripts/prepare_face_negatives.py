"""Download verified LFW archive and create an identity-disjoint face-negative subset."""
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.request
from PIL import Image

URL='https://ndownloader.figshare.com/files/5976015'
SHA256='b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a'
ROOT=Path(__file__).resolve().parents[1]

def main():
    output=ROOT/'hard_negatives/lfw'
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding='utf8'))
        if existing['source']!=URL or existing['archive_sha256']!=SHA256:
            raise ValueError('Existing subset has an unexpected source')
        for record in existing['records']:
            image=(output/record['path']).resolve()
            if not image.is_relative_to(output.resolve()) or hashlib.sha256(image.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Existing subset image changed or escaped its root')
        print('Existing verified subset retained; no download needed'); return
    archive=ROOT/'.cache/lfw-funneled.tgz'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        temp=archive.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=60) as source, temp.open('wb') as dest:
            total=0
            while chunk:=source.read(1024*1024):
                dest.write(chunk); total+=len(chunk)
                if total%(25*1024*1024)<1024*1024: print(f'Downloaded {total//1024//1024} MiB',flush=True)
        temp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('LFW archive checksum mismatch')
    output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        groups={}
        for entry in tar.getmembers():
            if entry.isfile() and entry.name.lower().endswith('.jpg'):
                groups.setdefault(Path(entry.name).parent.name,[]).append(entry)
        identities=sorted(groups); rng=random.Random(20261018); rng.shuffle(identities)
        records=[]; hashes=set()
        # One image per person: no person can occur in more than one split.
        selected=[]
        for i,identity in enumerate(identities[:600]):
            members=sorted(groups[identity],key=lambda m:m.name)
            selected.append((i,identity,rng.choice(members)))
        # Read archive order to avoid repeatedly decompressing a gzip stream.
        for i,identity,member in sorted(selected,key=lambda item:item[2].offset_data):
            data=tar.extractfile(member).read()
            digest=hashlib.sha256(data).hexdigest()
            if digest in hashes: raise ValueError('Duplicate image')
            hashes.add(digest)
            Image.open(io.BytesIO(data)).verify()
            split='train' if i<420 else 'val' if i<510 else 'test'
            name=f'{split}/{i:04d}.jpg'; path=output/name; path.parent.mkdir(exist_ok=True); path.write_bytes(data)
            records.append(dict(path=name,split=split,identity=identity,source_member=member.name,sha256=digest))
    manifest=dict(source=URL,archive_sha256=SHA256,source_documentation='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html',
                  purpose='Non-family real human face negatives; no identity prediction task',seed=20261018,records=records)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'images':len(records),'train':420,'val':90,'test':90,'identity_disjoint':True}),flush=True)

if __name__=='__main__':main()
