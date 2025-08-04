from collections import deque
from typing import List

import torch
import torch.nn.functional as F


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                   F.kl_div(q.log(), m, reduction='batchmean'))


class TIAWWeighting:
    """Temporal Instability-Aware Weighting following the paper."""

    def __init__(self, num_samples: int, num_classes: int, window_size: int = 5,
                 lambda_t: float = 0.5, omega_min: float = 0.5, beta: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        self.num_classes = num_classes
        self.device = device
        self.window_size = window_size
        self.lambda_t = lambda_t
        self.omega_min = omega_min
        self.beta = beta
        self.history: List[deque] = [deque(maxlen=window_size) for _ in range(num_samples)]

    def update_and_get_weights(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        entropies = -torch.sum(probs * probs.clamp_min(1e-8).log(), dim=1)
        jsd_list = []
        for idx, p in zip(indices.tolist(), probs):
            hist = list(self.history[idx])
            if len(hist) == 0:
                jsd = torch.zeros(1, device=self.device)
            else:
                jsd = torch.stack([js_divergence(p.unsqueeze(0), h.unsqueeze(0)) for h in hist]).mean()
            jsd_list.append(jsd)
        jsd_tensor = torch.stack(jsd_list).to(self.device)
        s = self.lambda_t * entropies + (1 - self.lambda_t) * jsd_tensor
        s_min = s.min()
        s_max = s.max()
        s_hat = (s - s_min) / (s_max - s_min + 1e-12)
        weights = 1 - (1 - self.omega_min) * torch.exp(-self.beta * s_hat)
        for idx, p in zip(indices.tolist(), probs):
            self.history[idx].append(p.detach())
        return weights.detach()