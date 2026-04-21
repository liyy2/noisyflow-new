import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.stage1.networks import ConditionalVAE, VelocityField
from noisyflow.stage1.training import (
    flow_matching_loss,
    sample_flow_euler,
    sample_vae,
    train_flow_stage1,
    train_vae_stage1,
    vae_loss,
)


class Stage1Tests(unittest.TestCase):
    def test_flow_matching_loss_scalar(self):
        torch.manual_seed(0)
        model = VelocityField(d=4, num_classes=3, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        x = torch.randn(5, 4)
        y = torch.tensor([0, 1, 2, 1, 0])
        loss = flow_matching_loss(model, x, y)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_flow_matching_loss_with_conditional(self):
        torch.manual_seed(0)
        model = VelocityField(
            d=4,
            num_classes=3,
            hidden=[8],
            time_emb_dim=8,
            label_emb_dim=8,
            cond_dim=2,
            cond_emb_dim=4,
        )
        x = torch.randn(6, 4)
        y = torch.tensor([0, 1, 2, 1, 0, 2])
        cond = torch.randn(6, 2)
        loss = flow_matching_loss(model, x, y, cond=cond)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sample_flow_euler_shape(self):
        torch.manual_seed(0)
        model = VelocityField(d=3, num_classes=2, hidden=[8], time_emb_dim=8, label_emb_dim=8)
        labels = torch.tensor([0, 1, 1, 0])
        samples = sample_flow_euler(model, labels, n_steps=5)
        self.assertEqual(samples.shape, (4, 3))
        self.assertFalse(samples.requires_grad)

    def test_vae_loss_scalar(self):
        torch.manual_seed(0)
        model = ConditionalVAE(d=4, num_classes=3, hidden=[8], latent_dim=3, label_emb_dim=8)
        x = torch.randn(5, 4)
        y = torch.tensor([0, 1, 2, 1, 0])
        loss = vae_loss(model, x, y)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sample_vae_shape(self):
        torch.manual_seed(0)
        model = ConditionalVAE(d=3, num_classes=2, hidden=[8], latent_dim=4, label_emb_dim=8)
        labels = torch.tensor([0, 1, 1, 0])
        samples = sample_vae(model, labels)
        self.assertEqual(samples.shape, (4, 3))
        self.assertFalse(samples.requires_grad)

    def test_train_flow_stage1_with_conditional_batches(self):
        torch.manual_seed(0)
        model = VelocityField(
            d=2,
            num_classes=3,
            hidden=[8],
            time_emb_dim=8,
            label_emb_dim=8,
            cond_dim=1,
            cond_emb_dim=4,
        )
        x = torch.randn(30, 2)
        y = torch.randint(0, 3, (30,))
        cond = torch.randn(30, 1)
        loader = DataLoader(TensorDataset(x, y, cond), batch_size=10, shuffle=True)
        stats = train_flow_stage1(model, loader, epochs=1, lr=1e-2, dp=None, device="cpu")
        self.assertIn("flow_loss", stats)

    def test_train_vae_stage1_with_conditional_batches(self):
        torch.manual_seed(0)
        model = ConditionalVAE(
            d=2,
            num_classes=3,
            hidden=[8],
            latent_dim=4,
            label_emb_dim=8,
            cond_dim=1,
            cond_emb_dim=4,
        )
        x = torch.randn(30, 2)
        y = torch.randint(0, 3, (30,))
        cond = torch.randn(30, 1)
        loader = DataLoader(TensorDataset(x, y, cond), batch_size=10, shuffle=True)
        stats = train_vae_stage1(model, loader, epochs=1, lr=1e-2, dp=None, device="cpu")
        self.assertIn("vae_loss", stats)
        self.assertIn("vae_recon_loss", stats)
        self.assertIn("vae_kl_loss", stats)


if __name__ == "__main__":
    unittest.main()
