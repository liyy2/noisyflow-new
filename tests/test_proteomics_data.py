import tempfile
import unittest

import numpy as np
import torch

from noisyflow.data.proteomics import make_federated_4i_proteomics


class ProteomicsDataTests(unittest.TestCase):
    def _write_h5ad(self, path: str) -> None:
        try:
            import anndata
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest("anndata not installed") from exc

        rng = np.random.default_rng(0)
        x = rng.normal(size=(240, 12)).astype(np.float32)
        drugs = np.array(["control"] * 160 + ["dasatinib"] * 80, dtype=object)
        adata = anndata.AnnData(X=x)
        adata.obs["drug"] = drugs
        adata.write_h5ad(path)

    def test_make_federated_4i_proteomics_kmeans_shapes(self):
        try:
            import sklearn  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest("scikit-learn not installed") from exc

        with tempfile.TemporaryDirectory() as td:
            h5ad_path = f"{td}/toy_4i.h5ad"
            self._write_h5ad(h5ad_path)

            client_datasets, target_ref, target_test = make_federated_4i_proteomics(
                path=h5ad_path,
                source_drug="control",
                target_drug="dasatinib",
                n_source_clients=4,
                source_size_per_client=0.5,
                target_ref_size=0.5,
                target_test_size=0.25,
                standardize=True,
                pca_dim=8,
                label_mode="kmeans",
                num_labels=5,
                seed=0,
            )

            self.assertEqual(len(client_datasets), 4)
            for ds in client_datasets:
                x, y = ds.tensors
                self.assertEqual(x.shape[1], 8)
                self.assertEqual(y.shape, (x.shape[0],))
                self.assertTrue(torch.isfinite(x).all())
                self.assertGreaterEqual(int(y.min().item()), 0)
                self.assertLess(int(y.max().item()), 5)

            x_ref, y_ref = target_ref.tensors
            x_test, y_test = target_test.tensors
            self.assertEqual(x_ref.shape[1], 8)
            self.assertEqual(x_test.shape[1], 8)
            self.assertEqual(y_ref.shape, (x_ref.shape[0],))
            self.assertEqual(y_test.shape, (x_test.shape[0],))
            self.assertGreater(int(x_ref.shape[0]), 0)
            self.assertGreater(int(x_test.shape[0]), 0)

    def test_make_federated_4i_proteomics_no_labels_single_class(self):
        with tempfile.TemporaryDirectory() as td:
            h5ad_path = f"{td}/toy_4i.h5ad"
            self._write_h5ad(h5ad_path)

            client_datasets, target_ref, target_test = make_federated_4i_proteomics(
                path=h5ad_path,
                label_mode="none",
                standardize=False,
                pca_dim=None,
                seed=0,
            )

            for ds in client_datasets:
                _, y = ds.tensors
                self.assertEqual(int(y.max().item()), 0)
            self.assertEqual(int(target_ref.tensors[1].max().item()), 0)
            self.assertEqual(int(target_test.tensors[1].max().item()), 0)


if __name__ == "__main__":
    unittest.main()
