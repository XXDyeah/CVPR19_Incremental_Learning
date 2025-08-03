import torch
from collections import deque


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * (p / m).log(), dim=-1) +
                   torch.sum(q * (q / m).log(), dim=-1))


class TIAWWeighting:
    """Temporal Instability-Aware Weighting.

    The module keeps a short history of the prediction distribution for each
    training sample and computes an instability score as described in the
    accompanying paper. The score is then normalised and mapped to a loss
    weight ``omega`` which can be used to rescale sample losses during
    training.
    """

    def __init__(self, num_samples: int, num_classes: int, window_size: int = 5,
                 device: torch.device = torch.device('cpu')):
        self.num_classes = num_classes
        self.device = device
        self.window_size = window_size
        self.history = [deque(maxlen=window_size) for _ in range(num_samples)]
        self.instability = torch.zeros(num_samples, device=device)
        self.weights = torch.ones(num_samples, device=device)

    def update_history(self, indices: torch.Tensor, probs: torch.Tensor):
        """Append the latest prediction ``probs`` for given ``indices``."""
        for idx, p in zip(indices.tolist(), probs):
            if idx < 0:
                # Negative indices correspond to synthetic samples for which we
                # do not maintain history.
                continue
            self.history[idx].append(p.detach().to(self.device))
        self._recompute_weights(indices)

    def _recompute_weights(self, indices: torch.Tensor):
        scores = []
        valid_idx = []
        for idx in indices.tolist():
            if idx < 0:
                scores.append(0.0)
                continue
            hist = list(self.history[idx])
            if len(hist) == 0:
                scores.append(0.0)
                continue
            p_curr = hist[-1]
            entropy = -(p_curr * p_curr.clamp_min(1e-8).log()).sum()
            mean_p = torch.stack(hist, 0).mean(0)
            jsd = torch.stack([js_divergence(h, mean_p) for h in hist]).mean()
            s = (entropy * jsd).item()
            self.instability[idx] = s
            scores.append(s)
            valid_idx.append(idx)
        if valid_idx:
            s_tensor = self.instability[valid_idx]
            s_min = s_tensor.min()
            s_max = s_tensor.max()
            if (s_max - s_min) > 1e-6:
                norm = (s_tensor - s_min) / (s_max - s_min)
            else:
                norm = torch.zeros_like(s_tensor)
            for i, idx in enumerate(valid_idx):
                self.weights[idx] = 1 - norm[i]

    def get_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """Return weights for the provided sample ``indices``."""
        w = []
        for idx in indices.tolist():
            if idx < 0:
                w.append(1.0)  # synthetic samples use default weight
            else:
                w.append(self.weights[idx].item())
        return torch.tensor(w, device=self.device)
