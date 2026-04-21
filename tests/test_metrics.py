import importlib.util
import unittest

import torch

from noisyflow.metrics import (
    label_silhouette_score,
    per_label_centroid_distances,
    rbf_mmd2,
    rbf_mmd2_multi_gamma,
    same_label_domain_mixing,
    sliced_w2_distance,
)


sklearn_spec = importlib.util.find_spec("sklearn")


class MetricsTests(unittest.TestCase):
    def test_rbf_mmd2_scalar_finite(self):
        torch.manual_seed(0)
        x = torch.randn(64, 5)
        y = torch.randn(48, 5)
        mmd2 = rbf_mmd2(x, y, gamma=1.0)
        self.assertEqual(mmd2.dim(), 0)
        self.assertTrue(torch.isfinite(mmd2).item())
        self.assertGreaterEqual(float(mmd2.item()), -1e-6)

    def test_rbf_mmd2_multi_gamma_len(self):
        torch.manual_seed(0)
        x = torch.randn(300, 3)
        y = torch.randn(200, 3)
        out = rbf_mmd2_multi_gamma(x, y, gammas=[0.5, 1.0, 2.0], max_samples=128, seed=0)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(out)).tolist()))

    def test_sliced_w2_distance_nonnegative(self):
        torch.manual_seed(0)
        x = torch.randn(256, 5)
        y = torch.randn(256, 5)
        d = sliced_w2_distance(x, y, num_projections=32, max_samples=128, seed=0)
        self.assertTrue(torch.isfinite(torch.tensor(d)).item())
        self.assertGreaterEqual(float(d), 0.0)

    def test_sliced_w2_distance_zero_on_identical(self):
        torch.manual_seed(0)
        x = torch.randn(128, 4)
        d = sliced_w2_distance(x, x.clone(), num_projections=64, max_samples=None, seed=0)
        self.assertLessEqual(abs(float(d)), 1e-6)

    @unittest.skipIf(sklearn_spec is None, "scikit-learn not installed")
    def test_label_silhouette_score_high_for_separated_labels(self):
        torch.manual_seed(0)
        x = torch.cat(
            [
                0.15 * torch.randn(40, 2) + torch.tensor([0.0, 0.0]),
                0.15 * torch.randn(40, 2) + torch.tensor([4.0, 0.0]),
                0.15 * torch.randn(40, 2) + torch.tensor([0.0, 4.0]),
            ],
            dim=0,
        )
        labels = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
        score = label_silhouette_score(x, labels)
        self.assertGreater(score, 0.8)

    @unittest.skipIf(sklearn_spec is None, "scikit-learn not installed")
    def test_label_silhouette_score_drops_when_labels_are_mixed(self):
        torch.manual_seed(0)
        x = torch.cat(
            [
                0.15 * torch.randn(40, 2) + torch.tensor([0.0, 0.0]),
                0.15 * torch.randn(40, 2) + torch.tensor([4.0, 0.0]),
                0.15 * torch.randn(40, 2) + torch.tensor([0.0, 4.0]),
            ],
            dim=0,
        )
        true_labels = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
        mixed_labels = torch.tensor(([0, 1, 2] * 40)[: x.shape[0]])
        self.assertLess(label_silhouette_score(x, mixed_labels), label_silhouette_score(x, true_labels))

    @unittest.skipIf(sklearn_spec is None, "scikit-learn not installed")
    def test_same_label_domain_mixing_low_when_domains_are_separated(self):
        torch.manual_seed(0)
        x = torch.cat(
            [
                0.10 * torch.randn(30, 2) + torch.tensor([0.0, 0.0]),
                0.10 * torch.randn(30, 2) + torch.tensor([2.5, 0.0]),
                0.10 * torch.randn(30, 2) + torch.tensor([10.0, 0.0]),
                0.10 * torch.randn(30, 2) + torch.tensor([12.5, 0.0]),
            ],
            dim=0,
        )
        labels = torch.tensor([0] * 60 + [1] * 60)
        domains = torch.tensor([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
        score = same_label_domain_mixing(x, labels, domains, n_neighbors=10)
        self.assertLess(score, 0.05)

    @unittest.skipIf(sklearn_spec is None, "scikit-learn not installed")
    def test_same_label_domain_mixing_high_when_domains_are_interleaved(self):
        torch.manual_seed(0)
        x = torch.cat(
            [
                0.12 * torch.randn(30, 2) + torch.tensor([0.0, 0.0]),
                0.12 * torch.randn(30, 2) + torch.tensor([0.0, 0.0]),
                0.12 * torch.randn(30, 2) + torch.tensor([6.0, 0.0]),
                0.12 * torch.randn(30, 2) + torch.tensor([6.0, 0.0]),
            ],
            dim=0,
        )
        labels = torch.tensor([0] * 60 + [1] * 60)
        domains = torch.tensor([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
        score = same_label_domain_mixing(x, labels, domains, n_neighbors=10)
        self.assertGreater(score, 0.35)

    @unittest.skipIf(sklearn_spec is None, "scikit-learn not installed")
    def test_same_label_domain_mixing_ignores_cross_label_neighbors(self):
        torch.manual_seed(0)
        x = torch.cat(
            [
                0.05 * torch.randn(30, 2) + torch.tensor([0.0, 0.0]),
                0.05 * torch.randn(30, 2) + torch.tensor([10.0, 0.0]),
                0.05 * torch.randn(30, 2) + torch.tensor([10.1, 0.0]),
                0.05 * torch.randn(30, 2) + torch.tensor([0.1, 0.0]),
            ],
            dim=0,
        )
        labels = torch.tensor([0] * 60 + [1] * 60)
        domains = torch.tensor([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
        score = same_label_domain_mixing(x, labels, domains, n_neighbors=10)
        self.assertLess(score, 0.1)

    def test_per_label_centroid_distances_shrink_when_shift_moves_toward_target(self):
        target_x = torch.cat(
            [
                0.05 * torch.randn(20, 2) + torch.tensor([0.0, 0.0]),
                0.05 * torch.randn(20, 2) + torch.tensor([5.0, 0.0]),
            ],
            dim=0,
        )
        target_labels = torch.tensor([0] * 20 + [1] * 20)
        raw_x = torch.cat(
            [
                0.05 * torch.randn(20, 2) + torch.tensor([2.5, 0.0]),
                0.05 * torch.randn(20, 2) + torch.tensor([7.5, 0.0]),
            ],
            dim=0,
        )
        transported_x = torch.cat(
            [
                0.05 * torch.randn(20, 2) + torch.tensor([0.5, 0.0]),
                0.05 * torch.randn(20, 2) + torch.tensor([5.5, 0.0]),
            ],
            dim=0,
        )
        labels = torch.tensor([0] * 20 + [1] * 20)
        raw = per_label_centroid_distances(raw_x, labels, target_x, target_labels)
        transported = per_label_centroid_distances(transported_x, labels, target_x, target_labels)
        self.assertLess(sum(transported.values()) / len(transported), sum(raw.values()) / len(raw))
        self.assertLess(transported[0], raw[0])
        self.assertLess(transported[1], raw[1])


if __name__ == "__main__":
    unittest.main()
