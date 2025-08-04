import random
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .vqvae import VQVAE


def cutmix(x1: torch.Tensor, x2: torch.Tensor, lam: float = 0.75) -> torch.Tensor:
    """Apply CutMix with a fixed ratio as described in the paper."""
    B, C, H, W = x1.shape
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = random.randint(0, W - cut_w)
    cy = random.randint(0, H - cut_h)
    mixed = x1.clone()
    mixed[:, :, cy:cy + cut_h, cx:cx + cut_w] = x2[:, :, cy:cy + cut_h, cx:cx + cut_w]
    return mixed


def _extract_features(model: nn.Module, dataset: Dataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return features and labels for ``dataset`` using ``model``."""
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            feat = nn.Sequential(*list(model.children())[:-1])(imgs).view(imgs.size(0), -1)
            feats.append(feat.cpu())
            labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


def _class_stats(features: torch.Tensor, labels: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Compute class-wise means and covariances."""
    means: Dict[int, torch.Tensor] = {}
    covs: Dict[int, torch.Tensor] = {}
    for c in labels.unique():
        feats_c = features[labels == c]
        means[int(c)] = feats_c.mean(0)
        centered = feats_c - means[int(c)]
        covs[int(c)] = (centered.t() @ centered) / max(1, feats_c.size(0) - 1)
    return means, covs


class CBAModule:
    """Counterfactual Boundary Augmentation implementation following the paper."""

    def __init__(self, num_classes: int, vqvae: VQVAE, k: int = 1, m: int = 20,
                 lambda_d: float = 0.5, lambda_s: float = 0.5, noise_std: float = 0.05,
                 mix_lam: float = 0.75, lambda_cba: float = 1.0, device: torch.device = torch.device("cpu")):
        self.num_classes = num_classes
        self.vqvae = vqvae
        self.k = k
        self.m = m
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.noise_std = noise_std
        self.mix_lam = mix_lam
        self.lambda_cba = lambda_cba
        self.device = device
        self.confusing_pairs: List[Tuple[int, int]] = []

    def select_pairs(self, mu_new: Dict[int, torch.Tensor], cov_new: Dict[int, torch.Tensor],
                     mu_old_prev: Dict[int, torch.Tensor], mu_old_curr: Dict[int, torch.Tensor],
                     cov_old_prev: Dict[int, torch.Tensor]):
        pairs: List[Tuple[Tuple[int, int], float]] = []
        for cn, mu_n in mu_new.items():
            for co, mu_o_prev in mu_old_prev.items():
                term1 = (mu_n - mu_o_prev).pow(2).sum()
                drift = (mu_old_curr[co] - mu_o_prev).pow(2).sum()
                blur = torch.trace(cov_new[cn]) + torch.trace(cov_old_prev[co])
                delta = term1 - self.lambda_d * drift - self.lambda_s * blur
                pairs.append(((cn, co), delta.item()))
        pairs.sort(key=lambda x: x[1])
        self.confusing_pairs = [p for p, _ in pairs[:self.k]]

    def _sample_by_class(self, dataset: Dataset, class_idx: int, num: int) -> List[torch.Tensor]:
        indices = [i for i in range(len(dataset)) if dataset[i][1] == class_idx]
        random.shuffle(indices)
        indices = indices[:num]
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
        imgs = []
        for img, _ in loader:
            imgs.append(img.to(self.device))
        return imgs

    def synthesize(self, x_n: torch.Tensor, x_o: torch.Tensor) -> torch.Tensor:
        x_mix = cutmix(x_o, x_n, lam=self.mix_lam)
        z_n, _ = self.vqvae.encode(x_n)
        z_o, _ = self.vqvae.encode(x_o)
        z_mix, _ = self.vqvae.encode(x_mix)
        alpha = torch.rand_like(z_n)
        z_cf = z_mix + alpha * (z_n - z_o) + torch.randn_like(z_n) * self.noise_std
        x_cf = self.vqvae.decode(z_cf)
        return x_cf

    def generate_dataset(self, model: nn.Module, ref_model: nn.Module,
                         dataset_new: Dataset, dataset_old: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract features
        feat_new, lbl_new = _extract_features(model, dataset_new, self.device)
        feat_old_curr, lbl_old = _extract_features(model, dataset_old, self.device)
        feat_old_prev, _ = _extract_features(ref_model, dataset_old, self.device) if ref_model is not None else (feat_old_curr, lbl_old)
        mu_new, cov_new = _class_stats(feat_new, lbl_new)
        mu_old_prev, cov_old_prev = _class_stats(feat_old_prev, lbl_old)
        mu_old_curr, _ = _class_stats(feat_old_curr, lbl_old)
        self.select_pairs(mu_new, cov_new, mu_old_prev, mu_old_curr, cov_old_prev)
        images = []
        labels = []
        for cn, co in self.confusing_pairs:
            imgs_n = self._sample_by_class(dataset_new, cn, self.m)
            imgs_o = self._sample_by_class(dataset_old, co, self.m)
            for xn, xo in zip(imgs_n, imgs_o):
                x_cf = self.synthesize(xn, xo)
                images.append(x_cf.squeeze(0).cpu())
                soft = torch.zeros(self.num_classes)
                soft[cn] = 0.5
                soft[co] = 0.5
                labels.append(soft)
        if images:
            return torch.stack(images), torch.stack(labels)
        return None, None

    def loss(self, logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        return (-soft * log_p).sum(dim=1)
