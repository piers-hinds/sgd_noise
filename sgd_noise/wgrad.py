import torch
import torch.nn as nn
import numpy as np


class WeightedGradient(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.alpha = 1
        for parameter in self.model.parameters():
            parameter.register_hook(lambda grad: grad * self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def weighted_backward(model: nn.Module, losses: torch.Tensor) -> None:
    for loss in losses:
        model.alpha = loss / (losses.norm() * np.sqrt(len(losses)))
        loss.backward(retain_graph=True)
