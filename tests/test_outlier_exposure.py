import unittest
import torch
from torch.utils.data import DataLoader,TensorDataset
from scripts.train_face_rejection import evaluate,outlier_exposure_loss

class OutlierExposureTests(unittest.TestCase):
    def test_unknown_loss_is_minimized_at_uniform_scores(self):
        self.assertAlmostEqual(outlier_exposure_loss(torch.zeros(3,11)).item(),0,places=6)
        sharp=torch.zeros(3,11);sharp[:,10]=8
        self.assertGreater(outlier_exposure_loss(sharp).item(),1)

    def test_unknown_gradient_reduces_spurious_winning_score(self):
        x=torch.tensor([[0.,0.,6.]],requires_grad=True)
        outlier_exposure_loss(x).backward()
        self.assertGreater(x.grad[0,2].item(),0)
        self.assertLess(x.grad[0,0].item(),0)

    def test_audit_distinguishes_top1_from_accepted_predictions(self):
        known=DataLoader(TensorDataset(torch.tensor([[6.,0.,0.],[0.,0.,6.]]),torch.tensor([0,2])),batch_size=2)
        # A negative can still rank dounai first while being correctly rejected.
        negative=DataLoader(torch.tensor([[0.,0.,0.1],[0.,0.,6.]]),batch_size=2)
        result=evaluate(torch.nn.Identity(),known,negative,torch.device('cpu'),2)
        self.assertEqual(result['known_accepted_correct'],2)
        self.assertEqual(result['negative_top1_dounai'],2)
        self.assertEqual(result['negative_false_accepted'],1)
        self.assertEqual(result['negative_accepted_as_dounai'],1)

if __name__=='__main__':unittest.main()
