"""Create a local-only audit page for the actual browser Canvas/WASM pipeline."""
from pathlib import Path
import argparse,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data_processed_v12')
    p.add_argument('--negatives',type=Path,default=ROOT/'hard_negatives/mixed_v12')
    a=p.parse_args();out=ROOT/'docs/verification-temp';out.mkdir(exist_ok=True);cases=[]
    negative=json.loads((a.negatives/'manifest.json').read_text(encoding='utf8'))
    for i,r in enumerate(r for r in negative['records'] if r['split']=='test'):
        name=f'negative-{i:03d}'+Path(r['path']).suffix;shutil.copyfile(a.negatives/r['path'],out/name)
        cases.append(dict(file=name,kind=r.get('source_kind','real_face')))
    known=json.loads((a.data/'manifest.json').read_text(encoding='utf8'))
    for i,r in enumerate(r for r in known['records'] if r['split']=='test'):
        name=f'known-{i:03d}.png';shutil.copyfile(a.data/r['path'],out/name)
        cases.append(dict(file=name,kind='known',expected=r['class']))
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf8')
    shutil.copyfile(ROOT/'tests/browser_model_audit.html',out/'index.html')
    print('Open http://127.0.0.1:8000/verification-temp/ after serving docs; cases:',len(cases))

if __name__=='__main__':main()
