"""Merge new synthetic training images while retaining legacy validation/test splits."""
from pathlib import Path
import argparse,copy,json,shutil
from collections import Counter
from PIL import Image,ImageOps
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preprocess import EXTENSIONS,image_digest,write_json
from common import load_manifest

def square_two_pixel_fix(image):
    w,h=image.size
    if abs(w-h)==2:
        if w>h:return image.crop((1,0,w-1,h)),True
        return image.crop((0,1,w,h-1)),True
    return image,False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'new_data')
    p.add_argument('--base',type=Path,default=ROOT/'data_processed')
    p.add_argument('--output',type=Path,default=ROOT/'data_processed_v12')
    a=p.parse_args();old=json.loads((a.base/'manifest.json').read_text(encoding='utf8'))
    missing=[r for r in old['records'] if not (a.base/r['path']).is_file()]
    if any(r['split']!='train' for r in missing):raise ValueError('Missing validation/test image')
    old['records']=[r for r in old['records'] if (a.base/r['path']).is_file()]
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty')
    folders={d.name:d for d in a.source.iterdir() if d.is_dir()}
    if set(folders)!=set(old['classes']):raise ValueError('New class directories do not match the existing classes')
    manifest=copy.deepcopy(old);manifest['parent_dataset']=str(a.base);manifest['augmentation_batch']='v1.2';manifest['missing_parent_training_records']=missing
    seen={r['sha256']:r for r in manifest['records']};duplicates=[];added=[];fixed=0
    a.output.mkdir(parents=True,exist_ok=True)
    for r in old['records']:
        target=a.output/r['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.base/r['path'],target)
    for cls,folder in sorted(folders.items()):
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        for file in files:
            with Image.open(file) as im:
                im=ImageOps.exif_transpose(im).convert('RGBA');bg=Image.new('RGBA',im.size,'white');im=Image.alpha_composite(bg,im).convert('RGB')
                original_size=list(im.size);im,fix=square_two_pixel_fix(im);fixed+=int(fix)
                # Newly generated images have no source watermark: no legacy bottom crop.
                im=ImageOps.pad(im,(old['preprocess']['size'],)*2,method=Image.Resampling.LANCZOS,color='white')
            digest=image_digest(im)
            if digest in seen:
                if seen[digest]['class']!=cls:raise ValueError('Cross-class duplicate')
                duplicates.append(dict(source=str(file),same_as=seen[digest]['path']));continue
            dest=Path('train')/cls/(digest[:20]+'.png');target=a.output/dest;target.parent.mkdir(parents=True,exist_ok=True);im.save(target)
            record=dict(source=str(file.relative_to(a.source)),path=dest.as_posix(),split='train',**{'class':cls},sha256=digest,
                        synthetic=True,reference_group_id=f'new_data_v12/{cls}',original_size=original_size,
                        two_pixel_crop=fix,preprocessing='EXIF/RGB; crop 1 pixel per long edge only when dimensions differ by 2; pad to 224; no watermark crop')
            seen[digest]=record;manifest['records'].append(record);added.append(record)
    manifest['new_batch_summary']=dict(source=str(a.source),added=len(added),two_pixel_crops=fixed,duplicates=duplicates,missing_parent_training_records=missing,split_policy='All new correlated synthetic images train-only; original val/test retained')
    write_json(a.output/'manifest.json',manifest);write_json(a.output/'preprocess_config.json',manifest['preprocess'])
    load_manifest(a.output)
    summary=dict(total=len(manifest['records']),splits=dict(Counter(r['split'] for r in manifest['records'])),**manifest['new_batch_summary'])
    write_json(ROOT/'artifacts/v12_preprocessing.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
