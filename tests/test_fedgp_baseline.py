import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.baselines.fedgp import train_fedgp_classifier


class FedGPBaselineTests(unittest.TestCase):
    def test_train_fedgp_classifier_smoke(self):
        torch.manual_seed(0)

        client1_x = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.5, -0.4],
                [-1.1, -0.7],
                [1.0, 0.8],
                [1.4, 0.6],
                [1.8, 1.0],
            ],
            dtype=torch.float32,
        )
        client1_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        client2_x = torch.tensor(
            [
                [-1.8, 0.3],
                [-1.4, 0.0],
                [-0.9, 0.2],
                [1.1, 1.1],
                [1.5, 1.0],
                [2.0, 0.7],
            ],
            dtype=torch.float32,
        )
        client2_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        target_ref_x = torch.tensor(
            [
                [-1.6, 0.1],
                [-1.0, 0.2],
                [1.2, 0.9],
                [1.6, 1.0],
            ],
            dtype=torch.float32,
        )
        target_ref_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        test_x = torch.tensor(
            [
                [-2.1, 0.0],
                [-1.2, 0.1],
                [-0.8, 0.3],
                [1.0, 0.8],
                [1.3, 1.0],
                [2.1, 0.9],
            ],
            dtype=torch.float32,
        )
        test_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        client_loaders = [
            DataLoader(TensorDataset(client1_x, client1_y), batch_size=3, shuffle=True, drop_last=False),
            DataLoader(TensorDataset(client2_x, client2_y), batch_size=3, shuffle=True, drop_last=False),
        ]
        target_loader = DataLoader(
            TensorDataset(target_ref_x, target_ref_y), batch_size=2, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=3, shuffle=False, drop_last=False)

        stats = train_fedgp_classifier(
            client_loaders,
            target_loader,
            test_loader,
            d=2,
            num_classes=2,
            hidden=[8],
            rounds=8,
            source_epochs=1,
            target_epochs=1,
            lr=5e-2,
            beta=0.5,
            device="cpu",
            name="Test/FedGP",
        )

        self.assertIn("acc", stats)
        self.assertIn("positive_projection_fraction", stats)
        self.assertGreaterEqual(stats["acc"], 0.0)
        self.assertLessEqual(stats["acc"], 1.0)
        self.assertGreater(stats["acc"], 0.8)


if __name__ == "__main__":
    unittest.main()
