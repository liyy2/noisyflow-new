import tempfile
import unittest

import numpy as np
import torch

from noisyflow.data.pamap2 import make_federated_pamap2


class Pamap2DataTests(unittest.TestCase):
    def _write_npz(self, path: str) -> None:
        rng = np.random.default_rng(0)
        d = 12

        n_subj_101 = 50
        n_subj_102 = 60
        n_subj_103 = 40
        x = rng.normal(size=(n_subj_101 + n_subj_102 + n_subj_103, d)).astype(np.float32)

        labels = rng.integers(0, 3, size=(x.shape[0],), endpoint=False).astype(np.int64)
        subjects = np.array([101] * n_subj_101 + [102] * n_subj_102 + [103] * n_subj_103, dtype=np.int64)
        np.savez(path, X=x, label=labels, subject=subjects)

    def test_make_federated_pamap2_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/pamap2_windows.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_pamap2(
                path=npz_path,
                target_subject=103,
                source_subjects=(101, 102),
                n_per_client=20,
                target_ref_size=10,
                target_test_size=10,
                standardize=False,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 2)
            for ds in client_datasets:
                x, y = ds.tensors
                self.assertEqual(x.shape, (20, 12))
                self.assertEqual(y.shape, (20,))
                self.assertTrue(torch.isfinite(x).all())

            x_ref, y_ref = target_ref.tensors
            x_test, y_test = target_test.tensors
            self.assertEqual(x_ref.shape, (10, 12))
            self.assertEqual(y_ref.shape, (10,))
            self.assertEqual(x_test.shape, (10, 12))
            self.assertEqual(y_test.shape, (10,))

    def test_make_federated_pamap2_pca(self):
        with tempfile.TemporaryDirectory() as td:
            npz_path = f"{td}/pamap2_windows.npz"
            self._write_npz(npz_path)

            client_datasets, target_ref, target_test = make_federated_pamap2(
                path=npz_path,
                target_subject=103,
                source_subjects=(101, 102),
                n_per_client=25,
                target_ref_size=0.25,
                target_test_size=0.25,
                standardize=True,
                pca_dim=4,
                seed=0,
            )

            self.assertEqual(client_datasets[0].tensors[0].shape[1], 4)
            self.assertEqual(target_ref.tensors[0].shape[1], 4)
            self.assertEqual(target_test.tensors[0].shape[1], 4)


if __name__ == "__main__":
    unittest.main()

