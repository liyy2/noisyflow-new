import tempfile
import unittest

import numpy as np
import torch

from noisyflow.data.camelyon17 import make_federated_camelyon17, make_federated_camelyon17_wilds


class Camelyon17DataTests(unittest.TestCase):
    def _write_npz(self, path: str) -> None:
        rng = np.random.default_rng(0)
        d = 8

        # Two source hospitals (0, 3) and one target hospital (2).
        n0 = 40
        n3 = 40
        n2 = 20
        x = rng.normal(size=(n0 + n3 + n2, d)).astype(np.float32)
        labels = rng.integers(0, 2, size=(n0 + n3 + n2,), endpoint=False).astype(np.int64)

        hospitals = np.array([0] * n0 + [3] * n3 + [2] * n2, dtype=np.int64)
        splits = np.array([0] * n0 + [1] * n3 + [2] * n2, dtype=np.int64)  # train, id_val, test

        np.savez(path, X=x, label=labels, hospital=hospitals, split=splits)

    def _write_npz_multi_target(self, path: str) -> None:
        rng = np.random.default_rng(0)
        d = 4

        # Two source hospitals (0, 1) and three target hospitals (2, 3, 4).
        n0, n1, n2, n3, n4 = 20, 20, 15, 15, 15

        x0 = np.full((n0, d), 0.0, dtype=np.float32)
        x1 = np.full((n1, d), 1.0, dtype=np.float32)
        x2 = np.full((n2, d), 2.0, dtype=np.float32)
        x3 = np.full((n3, d), 3.0, dtype=np.float32)
        x4 = np.full((n4, d), 4.0, dtype=np.float32)
        x = np.concatenate([x0, x1, x2, x3, x4], axis=0)

        n_total = n0 + n1 + n2 + n3 + n4
        labels = rng.integers(0, 2, size=(n_total,), endpoint=False).astype(np.int64)
        hospitals = np.array([0] * n0 + [1] * n1 + [2] * n2 + [3] * n3 + [4] * n4, dtype=np.int64)

        # train/id_val are source; test is target.
        splits = np.array([0] * n0 + [1] * n1 + [2] * n2 + [2] * n3 + [2] * n4, dtype=np.int64)
        np.savez(path, X=x, label=labels, hospital=hospitals, split=splits)

    def test_make_federated_camelyon17_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/camelyon17_small.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_camelyon17_wilds(
                path=npz_path,
                source_splits=("train", "id_val"),
                target_split="test",
                target_hospital=2,
                n_per_client=20,
                target_ref_size=5,
                target_test_size=10,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 2)
            for ds in client_datasets:
                x, y = ds.tensors
                self.assertEqual(x.shape, (20, 8))
                self.assertEqual(y.shape, (20,))
                self.assertTrue(torch.isfinite(x).all())

            x_ref, y_ref = target_ref.tensors
            x_test, y_test = target_test.tensors
            self.assertEqual(x_ref.shape, (5, 8))
            self.assertEqual(y_ref.shape, (5,))
            self.assertEqual(x_test.shape, (10, 8))
            self.assertEqual(y_test.shape, (10,))

    def test_make_federated_camelyon17_pca(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/camelyon17_small.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_camelyon17_wilds(
                path=npz_path,
                source_splits=(0, 1),
                target_split=2,
                target_hospital=2,
                n_per_client=10,
                target_ref_size=0.25,
                target_test_size=0.25,
                standardize=True,
                pca_dim=4,
                seed=0,
            )

            self.assertEqual(client_datasets[0].tensors[0].shape[1], 4)
            self.assertEqual(target_ref.tensors[0].shape[1], 4)
            self.assertEqual(target_test.tensors[0].shape[1], 4)

    def test_make_federated_camelyon17_multi_target_hospitals(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/camelyon17_multi_target.npz"
            self._write_npz_multi_target(npz_path)

            client_datasets, target_ref, target_test = make_federated_camelyon17_wilds(
                path=npz_path,
                source_splits=("train", "id_val"),
                target_split="test",
                source_hospitals=(0, 1),
                target_hospitals=(2, 3, 4),
                n_per_client=10,
                target_ref_size=10,
                target_test_size=10,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 2)
            source_vals = set()
            for ds in client_datasets:
                x, _ = ds.tensors
                source_vals |= set(torch.unique(x[:, 0]).tolist())
            target_vals = set(torch.unique(target_ref.tensors[0][:, 0]).tolist()) | set(
                torch.unique(target_test.tensors[0][:, 0]).tolist()
            )
            self.assertTrue(source_vals.issubset({0.0, 1.0}))
            self.assertTrue(target_vals.issubset({2.0, 3.0, 4.0}))
            self.assertTrue(source_vals.isdisjoint(target_vals))

    def test_make_federated_camelyon17_explicit_hospitals(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/camelyon17_explicit.npz"
            self._write_npz_multi_target(npz_path)

            client_datasets, target_ref, target_test = make_federated_camelyon17(
                path=npz_path,
                source_hospitals=(0, 1),
                target_hospitals=(2, 3, 4),
                n_per_client=10,
                target_ref_size=10,
                target_test_size=10,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 2)
            source_vals = set()
            for ds in client_datasets:
                x, _ = ds.tensors
                source_vals |= set(torch.unique(x[:, 0]).tolist())
            target_vals = set(torch.unique(target_ref.tensors[0][:, 0]).tolist()) | set(
                torch.unique(target_test.tensors[0][:, 0]).tolist()
            )
            self.assertTrue(source_vals.issubset({0.0, 1.0}))
            self.assertTrue(target_vals.issubset({2.0, 3.0, 4.0}))


if __name__ == "__main__":
    unittest.main()
