"""Prepare LFW + Google Cartoon Set negatives with disjoint sample groups."""
from pathlib import Path
import hashlib,json,random,tarfile,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL='https://huggingface.co/datasets/cgarciae/cartoonset/resolve/1.0.0/data/cartoonset10k.tgz'
SHA='902145ad5096af10ca9ee200b648e2a60e8b6b83dd2f4dc5356817988be8ff43'

def main():
    out=ROOT/'hard_negatives/mixed_v12'
    if (out/'manifest.json').exists():print('Mixed negatives already prepared');return
    archive=ROOT/'.cache/cartoonset10k.tgz';archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        temp=archive.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=60) as source,temp.open('wb') as dest:
            while chunk:=source.read(1024*1024):dest.write(chunk)
        temp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA:raise ValueError('Cartoon Set archive checksum mismatch')
    out.mkdir(parents=True,exist_ok=True);records=[]
    lfwroot=ROOT/'hard_negatives/lfw';lfw=json.loads((lfwroot/'manifest.json').read_text(encoding='utf8'))
    for r in lfw['records']:
        target=out/'lfw'/r['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(lfwroot/r['path'],target)
        records.append({**r,'path':'lfw/'+r['path'],'identity':'lfw/'+r['identity'],'source_kind':'real_face'})
    with tarfile.open(archive,'r:gz') as tar:
        members=sorted([m for m in tar.getmembers() if m.isfile() and m.name.endswith('.png')],key=lambda m:m.name)
        random.Random(20261019).shuffle(members);chosen=members[:600]
        indexed=list(enumerate(chosen))
        for i,m in sorted(indexed,key=lambda pair:pair[1].offset_data):
            data=tar.extractfile(m).read();split='train' if i<420 else 'val' if i<510 else 'test'
            name=f'cartoonset/{split}/{i:04d}.png';target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records.append(dict(path=name,split=split,identity='cartoonset/'+Path(m.name).stem,source_kind='cartoon_face',
                                source_member=m.name,sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(source=[lfw['source'],URL],archive_sha256={'lfw':lfw['archive_sha256'],'cartoonset':SHA},
                  attribution='Cartoon Set: Google LLC, CC BY 4.0; https://google.github.io/cartoonset/download.html',
                  records=records,seed=20261019)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'ATTRIBUTION.txt').write_text('Google Cartoon Set, Google LLC. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\nSource: https://google.github.io/cartoonset/download.html\nImages unmodified; 600-image subset. Training preprocessing and augmentation are applied in memory.\nLFW source and original archive hash are recorded in manifest.json.\n',encoding='utf8')
    print(json.dumps({'total':len(records),'train':840,'val':180,'test':180}))

if __name__=='__main__':main()
