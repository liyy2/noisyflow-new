"""Baselines for comparing NoisyFlow against prior work."""

from noisyflow.baselines.federated_classifier import (
    average_model_state_dicts,
    train_fedavg_classifier,
    train_fedavg_classifier_with_model,
)
from noisyflow.baselines.fedgp import train_fedgp_classifier, train_fedgp_classifier_with_model

__all__ = [
    "average_model_state_dicts",
    "train_fedavg_classifier",
    "train_fedavg_classifier_with_model",
    "train_fedgp_classifier",
    "train_fedgp_classifier_with_model",
]
