import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .vqvae import VQVAE


def cutmix(x1: torch.Tensor, x2: torch.Tensor, lam: float = 0.75) -> torch.Tensor:
    """Apply a simple CutMix operation with fixed area ratio.

    ``lam`` controls the portion of ``x1`` to keep. A value of 0.75 means 75%
    of ``x1`` is preserved while 25% is replaced by a patch from ``x2``.
    """
    B, C, H, W = x1.shape
    cut_w, cut_h = int(W * (1 - lam) ** 0.5), int(H * (1 - lam) ** 0.5)
    cx = random.randint(0, W - cut_w)
    cy = random.randint(0, H - cut_h)
    mixed = x1.clone()
    mixed[:, :, cy:cy + cut_h, cx:cx + cut_w] = x2[:, :, cy:cy + cut_h, cx:cx + cut_w]
    return mixed


def mahalanobis(mu1: torch.Tensor, mu2: torch.Tensor, cov_inv: torch.Tensor) -> float:
    diff = (mu1 - mu2).unsqueeze(0)
    return float(diff @ cov_inv @ diff.t())


class CBAModule:
    """Counterfactual Boundary Augmentation.

    This module synthesises counterfactual samples lying close to the decision
    boundary between confusing class pairs. The implementation follows the
    procedure described in the accompanying paper. A lightweight VQ-VAE is used
    for feature interpolation and decoding.
    """

    def __init__(self, num_classes: int, vqvae: VQVAE, k: int = 1,
                 noise_std: float = 0.05, lambda_cba: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        self.num_classes = num_classes
        self.vqvae = vqvae
        self.k = k
        self.noise_std = noise_std
        self.lambda_cba = lambda_cba
        self.device = device
        self.confusing_pairs: List[Tuple[int, int]] = []

    def update_statistics(self, features_new: torch.Tensor, labels_new: torch.Tensor,
                          features_old: torch.Tensor, labels_old: torch.Tensor):
        """Compute class means and select the most confusing class pairs.

        The confusion between a new class ``c_n`` and an old class ``c_o`` is
        measured using the Mahalanobis distance of their feature means as in
        Equation (1) of the paper.
        """
        all_features = torch.cat([features_new, features_old], dim=0)
        cov = torch.from_numpy(torch.cov(all_features.t().cpu())).float().to(self.device)
        cov_inv = torch.inverse(cov + torch.eye(cov.size(0), device=self.device) * 1e-5)
        means_new = {}
        for c in labels_new.unique():
            means_new[int(c)] = features_new[labels_new == c].mean(0)
        means_old = {}
        for c in labels_old.unique():
            means_old[int(c)] = features_old[labels_old == c].mean(0)
        pairs = []
        for cn, mu_n in means_new.items():
            for co, mu_o in means_old.items():
                d = mahalanobis(mu_n, mu_o, cov_inv)
                pairs.append(((int(cn), int(co)), d))
        pairs.sort(key=lambda x: x[1])
        self.confusing_pairs = [p for p, _ in pairs[:self.k]]

    def synthesize_pair(self, x_n: torch.Tensor, x_o: torch.Tensor,
                        c_n: int, c_o: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single counterfactual sample for the class pair."""
        x_mix = cutmix(x_n, x_o)
        z_n, _ = self.vqvae.encode(x_n)
        z_o, _ = self.vqvae.encode(x_o)
        z_interp = 0.5 * (z_n + z_o)
        z_adv = z_interp + torch.randn_like(z_interp) * self.noise_std
        x_adv = self.vqvae.decode(z_adv)
        soft = torch.zeros((1, self.num_classes), device=self.device)
        soft[0, c_n] = 0.5
        soft[0, c_o] = 0.5
        return x_adv, soft

    def generate(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Generate counterfactual samples for the current batch.

        For simplicity we randomly pick one confusing pair and synthesise a
        sample if both classes appear in the batch. The returned indices are
        set to -1 because synthetic samples are not tracked by TIAW.
        """
        if not self.confusing_pairs:
            return None, None, None
        cn, co = random.choice(self.confusing_pairs)
        if cn not in targets or co not in targets:
            return None, None, None
        idx_n = (targets == cn).nonzero()[0]
        idx_o = (targets == co).nonzero()[0]
        x_n = inputs[idx_n:idx_n + 1]
        x_o = inputs[idx_o:idx_o + 1]
        adv_x, soft = self.synthesize_pair(x_n, x_o, cn, co)
        adv_idx = torch.full((1,), -1, dtype=torch.long, device=self.device)
        return adv_x, soft, adv_idx

    def loss(self, logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        """Classification loss for counterfactual samples (Eq. 6)."""
        log_p = F.log_softmax(logits, dim=1)
        return (-soft * log_p).sum(dim=1)
