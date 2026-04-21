from __future__ import annotations

from noisyflow.data.brainscope import make_federated_brainscope
from noisyflow.data.camelyon17 import make_federated_camelyon17, make_federated_camelyon17_wilds
from noisyflow.data.cell import (
    make_cellot_lupuspatients_kang_hvg,
    make_cellot_sciplex3_hvg,
    make_cellot_statefate_invitro_hvg,
    make_federated_cell_dataset,
)
from noisyflow.data.pamap2 import make_federated_pamap2
from noisyflow.data.proteomics import make_federated_4i_proteomics
from noisyflow.data.synthetic import make_federated_mixture_gaussians
from noisyflow.data.toy import make_toy_federated_gaussians

__all__ = [
    "make_federated_brainscope",
    "make_federated_camelyon17",
    "make_federated_camelyon17_wilds",
    "make_cellot_lupuspatients_kang_hvg",
    "make_cellot_sciplex3_hvg",
    "make_cellot_statefate_invitro_hvg",
    "make_federated_cell_dataset",
    "make_federated_pamap2",
    "make_federated_4i_proteomics",
    "make_federated_mixture_gaussians",
    "make_toy_federated_gaussians",
]
