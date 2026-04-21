import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from noisyflow.baselines.federated_classifier import average_model_state_dicts, train_fedavg_classifier


class FederatedClassifierTests(unittest.TestCase):
    def test_average_model_state_dicts_weighted(self):
        state_a = {
            "weight": torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32),
            "counter": torch.tensor(2, dtype=torch.int64),
        }
        state_b = {
            "weight": torch.tensor([[3.0, 5.0], [7.0, 9.0]], dtype=torch.float32),
            "counter": torch.tensor(10, dtype=torch.int64),
        }

        averaged = average_model_state_dicts([state_a, state_b], weights=[1.0, 3.0])

        expected = torch.tensor([[2.5, 4.5], [6.5, 8.5]], dtype=torch.float32)
        self.assertTrue(torch.allclose(averaged["weight"], expected))
        self.assertEqual(int(averaged["counter"].item()), 2)

    def test_train_fedavg_classifier_smoke(self):
        torch.manual_seed(0)

        client1_x = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [-1.2, -0.8],
                [1.2, 0.8],
                [1.5, 0.5],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        client1_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        client2_x = torch.tensor(
            [
                [-2.2, 0.1],
                [-1.6, 0.5],
                [-1.1, 0.2],
                [1.1, 1.2],
                [1.7, 1.0],
                [2.1, 0.6],
            ],
            dtype=torch.float32,
        )
        client2_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        ref_x = torch.tensor([[-1.4, 0.0], [1.4, 0.9]], dtype=torch.float32)
        ref_y = torch.tensor([0, 1], dtype=torch.long)

        test_x = torch.tensor(
            [
                [-2.1, -0.2],
                [-1.3, 0.1],
                [-1.0, -0.4],
                [1.0, 0.7],
                [1.4, 1.1],
                [2.2, 0.9],
            ],
            dtype=torch.float32,
        )
        test_y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        client_loaders = [
            DataLoader(TensorDataset(client1_x, client1_y), batch_size=3, shuffle=True, drop_last=False),
            DataLoader(TensorDataset(client2_x, client2_y), batch_size=3, shuffle=True, drop_last=False),
        ]
        ref_loader = DataLoader(TensorDataset(ref_x, ref_y), batch_size=2, shuffle=True, drop_last=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=3, shuffle=False, drop_last=False)

        stats = train_fedavg_classifier(
            client_loaders,
            test_loader,
            d=2,
            num_classes=2,
            hidden=[8],
            epochs=10,
            lr=5e-2,
            ref_finetune_loader=ref_loader,
            ref_finetune_epochs=4,
            ref_finetune_lr=5e-2,
            device="cpu",
            name="Test/FedAvg",
        )

        self.assertIn("acc", stats)
        self.assertIn("client_count", stats)
        self.assertEqual(int(stats["client_count"]), 2)
        self.assertGreaterEqual(stats["acc"], 0.0)
        self.assertLessEqual(stats["acc"], 1.0)
        self.assertGreater(stats["acc"], 0.8)


if __name__ == "__main__":
    unittest.main()
