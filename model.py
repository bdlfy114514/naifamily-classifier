"""ResNet18 迁移学习 CNN；保留原有轻量 CNN 以兼容旧权重。"""
from pathlib import Path

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class NaiFamilyCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = .4):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes 至少为 2')
        self.features = nn.Sequential(
            ConvBlock(3, 24), ConvBlock(24, 48),
            ConvBlock(48, 96), ConvBlock(96, 160),
        )
        # 保留粗略空间信息，以区分耳朵、尾巴和身体轮廓。
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(160*4, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))  # 原始 logits，交叉熵内部处理 softmax


def build_model(num_classes, dropout=.2, arch='scratch', pretrained=False):
    """推理默认不下载权重；旧 checkpoint 无 arch 字段时仍构建原始网络。"""
    if arch == 'scratch':
        return NaiFamilyCNN(num_classes, dropout)
    if arch != 'resnet18':
        raise ValueError(f'未知网络结构：{arch}')
    if num_classes < 2 or not 0 <= dropout < 1:
        raise ValueError('类别数须至少为 2，dropout 须在 [0, 1)')
    from torchvision.models import ResNet18_Weights, resnet18
    if pretrained:
        # 权重缓存在项目内，便于离线重复实验。
        torch.hub.set_dir(str(Path(__file__).parent/'.cache'/'torch'))
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    network.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(network.fc.in_features, num_classes))
    return network


def configure_transfer_stage(network, warmup, unfreeze='last2'):
    """warmup 只训练分类头；之后按配置微调主干。"""
    for name, parameter in network.named_parameters():
        parameter.requires_grad = (name.startswith('fc.') or
                                   (not warmup and (unfreeze == 'all' or
                                    name.startswith(('layer3.', 'layer4.')))))


def freeze_batch_norm(network):
    """小数据微调时保留 ImageNet running mean/variance；affine 参数仍可学习。"""
    for module in network.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


if __name__ == '__main__':
    network = build_model(11, arch='resnet18').eval()
    with torch.inference_mode():
        print('output:', tuple(network(torch.zeros(2, 3, 224, 224)).shape))
    print('parameters:', sum(p.numel() for p in network.parameters()))
