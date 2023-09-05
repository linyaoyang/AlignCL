import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Any, Tuple


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient reverse layer with warm start"""

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.0,
                 max_iter: Optional[int] = 1000, auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iter = max_iter
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = np.float(2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number by 1"""
        self.iter_num += 1


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iter=10000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pred_s, pred_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pred_t, 'nuc') / y_t.shape[0] + torch.norm(pred_s, 'nuc') / y_s.shape[0])
        return loss

    def forward(self, f: torch.Tensor, source_idx: torch.LongTensor, target_idx: torch.LongTensor) -> torch.Tensor:
        feature_grl = self.grl(f)
        y = self.classifier(feature_grl)
        y_s, y_t = y[source_idx], y[target_idx]
        loss = self.n_discrepancy(y_s, y_t)
        return loss