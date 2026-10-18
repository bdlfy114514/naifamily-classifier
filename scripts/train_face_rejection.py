"""Fine-tune known classes with LFW outlier exposure; select using validation only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import copy
import json
import math
import random
import hashlib
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from common import ManifestDataset,build_transform,load_manifest,seed_everything,select_device
from model import build_model,configure_transfer_stage,freeze_batch_norm
from preprocess import prepare_image,write_json
from train import plot_history

def outlier_exposure_loss(logits):
    """KL(uniform || predicted), averaged across unknown examples."""
    return -nn.functional.log_softmax(logits.float(),dim=1).mean()-math.log(logits.shape[1])

class FaceNegatives(Dataset):
    def __init__(self,root,manifest,split,preprocess,normalization,training=False):
        self.records=[r for r in manifest['records'] if r['split']==split]
        self.images=[prepare_image(root/r['path'],preprocess) for r in self.records]
        self.transform=build_transform(training,**normalization,augmentation='gentle')
    def __len__(self):return len(self.images)
    def __getitem__(self,i):return self.transform(self.images[i])

def validate_negatives(root,manifest):
    seen=set(); identities={s:set() for s in ('train','val','test')}
    for r in manifest['records']:
        p=(root/r['path']).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():raise ValueError('Invalid negative image path')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=r['sha256'] or digest in seen:raise ValueError('Negative image changed or duplicated')
        seen.add(digest); identities[r['split']].add(r['identity'])
    if any(not v for v in identities.values()):raise ValueError('Empty negative split')
    if any(identities[a]&identities[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Identity leakage between negative splits')

@torch.inference_mode()
def evaluate(model,known,negative,device,dounai_index,ratio=1.5):
    model.eval(); probs=[]; labels=[]; neg_probs=[]
    for x,y in known:
        probs.append(model(x.to(device)).softmax(1).cpu());labels.append(y)
    for x in negative:neg_probs.append(model(x.to(device)).softmax(1).cpu())
    p=torch.cat(probs); y=torch.cat(labels); n=torch.cat(neg_probs)
    top=p.topk(2,dim=1); ntop=n.topk(2,dim=1)
    accept=top.values[:,0]>ratio*top.values[:,1]; na=ntop.values[:,0]>ratio*ntop.values[:,1]
    pred=top.indices[:,0]; npred=ntop.indices[:,0]; mask=y==dounai_index
    kcorrect=int(((pred==y)&accept).sum()); nreject=int((~na).sum())
    return dict(known_total=len(y),known_raw_correct=int((pred==y).sum()),known_accepted_correct=kcorrect,
                known_rejected=int((~accept).sum()),dounai_total=int(mask.sum()),
                dounai_raw_correct=int(((pred==y)&mask).sum()),dounai_accepted_correct=int(((pred==y)&mask&accept).sum()),
                negative_total=len(n),negative_false_accepted=int(na.sum()),negative_top1_dounai=int((npred==dounai_index).sum()),
                negative_accepted_as_dounai=int(((npred==dounai_index)&na).sum()),
                balanced_score=(kcorrect/len(y)+nreject/len(n))/2,
                known_cross_entropy=float(nn.functional.nll_loss(p.clamp_min(1e-30).log(),y)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ROOT/'runs/resnet18_transfer_v1/best.pt')
    p.add_argument('--negatives',type=Path,default=ROOT/'hard_negatives/lfw')
    p.add_argument('--output',type=Path,default=ROOT/'runs/resnet18_face_oe_v1')
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--oe-weight',type=float,default=1.)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=2e-4)
    p.add_argument('--device',choices=['auto','cpu','cuda','xpu'],default='auto')
    args=p.parse_args()
    if args.epochs<1 or args.oe_weight<=0 or min(args.lr,args.head_lr)<=0:p.error('Invalid hyperparameters')
    if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    seed_everything(20261018);torch.set_num_threads(2);device=select_device(args.device)
    ckpt=torch.load(args.weights,map_location='cpu',weights_only=True)
    data=ROOT/'data_processed'; manifest=load_manifest(data)
    if manifest!=ckpt['manifest']:raise ValueError('Known dataset split changed')
    nm=json.loads((args.negatives/'manifest.json').read_text(encoding='utf8'));validate_negatives(args.negatives,nm)
    normalization=ckpt['normalization'];prep=ckpt['preprocess'];classes=ckpt['classes']
    def kl(split,training=False):return DataLoader(ManifestDataset(data,manifest,split,build_transform(training,**normalization,augmentation='gentle')),batch_size=16,shuffle=training)
    def nl(split,training=False):return DataLoader(FaceNegatives(args.negatives,nm,split,prep,normalization,training),batch_size=16,shuffle=training)
    train,negtrain=kl('train',True),nl('train',True);val,nval=kl('val'),nl('val')
    model=build_model(**ckpt['model_config']);model.load_state_dict(ckpt['model_state']);model.to(device)
    configure_transfer_stage(model,False,'last2')
    groups=[{'params':[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('fc.')],'lr':args.lr},
            {'params':model.fc.parameters(),'lr':args.head_lr}]
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    amp=device.type in ('cuda','xpu');scaler=torch.amp.GradScaler(device.type,enabled=amp)
    baseline=evaluate(model,val,nval,device,classes.index('豆奶'))
    write_json(args.output/'baseline_validation.json',baseline)
    print('BASELINE VAL '+json.dumps(baseline),flush=True)
    best_score=(baseline['balanced_score'],baseline['known_raw_correct'],-baseline['known_cross_entropy'])
    history=[];best_epoch=0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(seed=20261018,method='outlier_exposure_uniform_cross_entropy',device_used=str(device),min_top_score_ratio=1.5,
                  selection='validation balanced mean of known accepted-correct rate and unknown rejection rate',
                  negatives_source=nm['source'],negatives_archive_sha256=nm['archive_sha256'])
    write_json(args.output/'config.json',config);write_json(args.output/'negative_manifest.json',nm)
    def save(epoch,metrics):
        checkpoint=copy.copy(ckpt)
        checkpoint.update(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          epoch=epoch,val_accuracy=metrics['known_raw_correct']/metrics['known_total'],
                          val_loss=metrics['known_cross_entropy'],training_config=config,rejection_validation=metrics,
                          parent_weights=str(args.weights),method='outlier_exposure')
        torch.save(checkpoint,args.output/'best.pt')
    save(0,baseline)
    for epoch in range(1,args.epochs+1):
        model.train();freeze_batch_norm(model); negiter=iter(negtrain);total=0;loss_sum=0
        for x,y in train:
            try:negative=next(negiter)
            except StopIteration:negiter=iter(negtrain);negative=next(negiter)
            x,y,negative=x.to(device),y.to(device),negative.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits=model(torch.cat([x,negative]));pos,neg=logits[:len(y)],logits[len(y):]
                known_loss=nn.functional.cross_entropy(pos,y,label_smoothing=.05)
                # Minimized at a uniform distribution: no known class should dominate unknown faces.
                oe_loss=outlier_exposure_loss(neg)
                loss=known_loss+args.oe_weight*oe_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(opt);scaler.update();loss_sum+=loss.item()*len(y);total+=len(y)
        scheduler.step();metrics=evaluate(model,val,nval,device,classes.index('豆奶'))
        score=(metrics['balanced_score'],metrics['known_raw_correct'],-metrics['known_cross_entropy'])
        row=dict(epoch=epoch,train_loss=loss_sum/total,val_loss=metrics['known_cross_entropy'],**metrics)
        history.append(row);write_json(args.output/'history.json',history)
        if score>best_score:best_score=score;best_epoch=epoch;save(epoch,metrics)
        print(f'EPOCH {epoch} loss={loss_sum/total:.4f} known={metrics["known_raw_correct"]}/33 accepted={metrics["known_accepted_correct"]}/33 neg_accept={metrics["negative_false_accepted"]}/90 score={metrics["balanced_score"]:.4f} best={best_epoch}',flush=True)
    plot_history(history,args.output/'loss_curve.png')
    # Held-out tests run only after validation-only model selection is complete.
    best=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best['model_state'])
    test,ntest=kl('test'),nl('test')
    candidate=evaluate(model,test,ntest,device,classes.index('豆奶'))
    model.load_state_dict(ckpt['model_state']);baseline_test=evaluate(model,test,ntest,device,classes.index('豆奶'))
    report=dict(best_epoch=best_epoch,baseline_validation=baseline,best_validation=best['rejection_validation'],
                baseline_test=baseline_test,candidate_test=candidate,
                limitations='LFW real faces only; no claim about cartoon faces, full-body images or all unknowns. Canvas/Pillow may differ.')
    write_json(args.output/'evaluation.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
