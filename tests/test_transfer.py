"""CPU 回归检查：旧模型兼容、微调冻结范围和 BatchNorm 统计量。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import NaiFamilyCNN, build_model, configure_transfer_stage
from train import run_epoch


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_configuration_keeps_old_state_dict(self):
        original = NaiFamilyCNN(11, .4)
        restored = build_model(num_classes=11, dropout=.4)
        restored.load_state_dict(original.state_dict(), strict=True)
        restored.eval()
        with torch.inference_mode():
            self.assertEqual(restored(torch.zeros(2, 3, 224, 224)).shape, (2, 11))

    def test_inference_construction_needs_no_download(self):
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('Unexpected download')):
            network = build_model(11, arch='resnet18').eval()
            with torch.inference_mode():
                logits = network(torch.zeros(2, 3, 224, 224))
            self.assertEqual(logits.shape, (2, 11))
            self.assertTrue(torch.isfinite(logits).all())

    def test_head_warmup_changes_only_head_and_preserves_bn(self):
        torch.manual_seed(42)
        network = build_model(11, arch='resnet18')
        configure_transfer_stage(network, warmup=True)
        backbone_before = network.conv1.weight.detach().clone()
        head_before = network.fc[1].weight.detach().clone()
        bn_before = network.bn1.running_mean.clone()
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.001)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        run_epoch(network, loader, nn.CrossEntropyLoss(), torch.device('cpu'), optimizer, scaler, freeze_bn=True)
        self.assertTrue(torch.equal(backbone_before, network.conv1.weight))
        self.assertTrue(torch.equal(bn_before, network.bn1.running_mean))
        self.assertFalse(torch.equal(head_before, network.fc[1].weight))
        self.assertIsNone(network.conv1.weight.grad)
        self.assertIsNotNone(network.fc[1].weight.grad)
        configure_transfer_stage(network, warmup=False, unfreeze='last2')
        self.assertFalse(network.layer2[0].conv1.weight.requires_grad)
        self.assertTrue(network.layer3[0].conv1.weight.requires_grad)
        self.assertTrue(network.layer4[0].conv1.weight.requires_grad)
        configure_transfer_stage(network, warmup=False, unfreeze='all')
        self.assertTrue(all(parameter.requires_grad for parameter in network.parameters()))


if __name__ == '__main__':
    unittest.main()
