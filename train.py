"""CNN 训练：ResNet18 两阶段微调，兼容旧版 scratch CNN；支持 CUDA/XPU/CPU。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (MEAN, STD, ManifestDataset, build_transform, load_manifest,
                    seed_everything, seed_worker, select_device)
from model import build_model, configure_transfer_stage, freeze_batch_norm
from preprocess import write_json


def plot_history(history, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train loss')
    ax.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='Validation loss')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Nai Family CNN')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, freeze_bn=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_batch_norm(model)
    total_loss, total_correct, total = 0., 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == 'cuda')
        labels = labels.to(device, non_blocking=device.type == 'cuda')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError('出现非有限 loss；可使用 --no-amp 或降低 --lr 后重试')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                scaler.step(optimizer)
                scaler.update()
        count = labels.size(0)
        total_loss += loss.detach().item()*count
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += count
    return total_loss/total, total_correct/total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).parent/'data_processed_v12')
    p.add_argument('--output', type=Path, help='默认 runs/时间戳；须为空目录')
    p.add_argument('--device', choices=['auto', 'cuda', 'xpu', 'cpu'], default='auto')
    p.add_argument('--arch', choices=['resnet18', 'scratch'], default='resnet18')
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                   help='ResNet18 首次训练下载官方 ImageNet 权重，之后使用项目缓存')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0, help='Windows 默认 0，避免多进程启动开销')
    p.add_argument('--lr', type=float, help='主干学习率，默认 ResNet18=1e-4，scratch=1e-3')
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--dropout', type=float, help='默认 ResNet18=.2，scratch=.4')
    p.add_argument('--warmup-epochs', type=int, default=3, help='预训练模型只训练分类头的轮数')
    p.add_argument('--unfreeze', choices=['last2', 'all'], default='last2')
    p.add_argument('--freeze-bn', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--label-smoothing', type=float, default=.05)
    p.add_argument('--augmentation', choices=['gentle', 'standard'], default='gentle')
    p.add_argument('--selection', choices=['accuracy', 'loss'], default='accuracy',
                   help='默认按验证准确率选最优，同分时按验证 loss；从不按测试集选择')
    p.add_argument('--patience', type=int, default=12, help='验证选择指标连续不改善时提前停止')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.arch == 'resnet18' else 1e-3
    if args.dropout is None:
        args.dropout = .2 if args.arch == 'resnet18' else .4
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        p.error('epochs/batch-size/patience 必须 >=1；workers 必须 >=0')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        p.error('lr 须为正数，weight-decay 须非负，dropout 须在 [0,1)')
    if not math.isfinite(args.head_lr) or args.head_lr <= 0 or not 0 <= args.label_smoothing < 1 or args.warmup_epochs < 0:
        p.error('head-lr 须为正数，label-smoothing 须在 [0,1)，warmup-epochs 须非负')
    device = select_device(args.device)
    seed_everything(args.seed)
    manifest = load_manifest(args.data)
    output = args.output or Path(__file__).parent/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists() and any(output.iterdir()):
        p.error('输出目录非空；请选择新的 --output')
    output.mkdir(parents=True, exist_ok=True)
    amp = args.amp and device.type in ('cuda', 'xpu')
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                       persistent_workers=args.workers > 0)
    train_set = ManifestDataset(args.data, manifest, 'train', build_transform(True, augmentation=args.augmentation))
    val_set = ManifestDataset(args.data, manifest, 'val', build_transform())
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    pretrained = args.arch == 'resnet18' and args.pretrained
    model = build_model(len(manifest['classes']), args.dropout, args.arch, pretrained=pretrained).to(device)
    if args.arch == 'resnet18':
        head = list(model.fc.parameters())
        head_ids = {id(parameter) for parameter in head}
        groups = [{'params': [parameter for parameter in model.parameters() if id(parameter) not in head_ids],
                   'lr': args.lr, 'name': 'backbone'},
                  {'params': head, 'lr': args.head_lr, 'name': 'head'}]
    else:
        groups = [{'params': list(model.parameters()), 'lr': args.lr, 'name': 'scratch'}]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(device_used=str(device), amp_used=amp, torch_version=str(torch.__version__),
                  classes=manifest['classes'], train_size=len(train_set), val_size=len(val_set),
                  pretrained_used=pretrained, scheduler='cosine', freeze_bn_used=pretrained and args.freeze_bn)
    write_json(output/'config.json', config)
    write_json(output/'manifest.json', manifest)
    print(f'Device: {device} | AMP: {amp} | train: {len(train_set)} | val: {len(val_set)}')
    print(f'Output: {output.resolve()}')
    history, best_score, stale, best_record = [], None, 0, None
    started = time.perf_counter()
    try:
        for epoch in range(1, args.epochs+1):
            epoch_started = time.perf_counter()
            warmup = pretrained and epoch <= args.warmup_epochs
            if pretrained:
                configure_transfer_stage(model, warmup, args.unfreeze)
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp,
                                              freeze_bn=pretrained and args.freeze_bn)
            val_loss, val_acc = run_epoch(model, val_loader, val_criterion, device, amp=amp)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       train_accuracy=train_acc, val_accuracy=val_acc, lr=optimizer.param_groups[0]['lr'],
                       head_lr=optimizer.param_groups[-1]['lr'], phase='head' if warmup else 'finetune',
                       seconds=time.perf_counter()-epoch_started)
            history.append(row)
            scheduler.step()
            score = (val_acc, -val_loss) if args.selection == 'accuracy' else (-val_loss,)
            if best_score is None or score > best_score:
                best_score, stale, best_record = score, 0, row.copy()
                # 保存 CPU Tensor，便于在另一后端加载；不保存 Python 模型对象。
                torch.save({'model_state': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'model_config': {'num_classes': len(manifest['classes']), 'dropout': args.dropout, 'arch': args.arch},
                            'classes': manifest['classes'], 'class_to_idx': manifest['class_to_idx'],
                            'preprocess': manifest['preprocess'], 'normalization': {'mean': MEAN, 'std': STD},
                            'epoch': epoch, 'val_loss': val_loss, 'val_accuracy': val_acc,
                            'manifest': manifest, 'training_config': config}, output/'best.pt.tmp')
                (output/'best.pt.tmp').replace(output/'best.pt')
            else:
                stale += 1
            with (output/'history.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            print(f'{epoch:03d}/{args.epochs} train loss={train_loss:.4f} acc={train_acc:.2%} | '
                  f'val loss={val_loss:.4f} acc={val_acc:.2%} | {row["phase"]} | {row["seconds"]:.1f}s', flush=True)
            if stale >= args.patience:
                print(f'Early stopping: {args.patience} epochs without validation {args.selection} improvement.')
                break
    finally:
        if history:
            plot_history(history, output/'loss_curve.png')
    write_json(output/'summary.json', {'best': best_record, 'selection': args.selection,
                                      'epochs_completed': len(history), 'seconds': time.perf_counter()-started,
                                      'test_used_for_selection': False})
    print(f'Best validation: epoch={best_record["epoch"]}, acc={best_record["val_accuracy"]:.2%}, '
          f'loss={best_record["val_loss"]:.4f}\nWeights: {output/"best.pt"}\nLoss plot: {output/"loss_curve.png"}')


if __name__ == '__main__':
    main()
