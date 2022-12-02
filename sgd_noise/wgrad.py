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


def backward_with_rejection(losses: torch.Tensor) -> None:
    """Takes individual losses, rejects with probability proportional to loss then computes the mean of the accepted
    losses and backpropogates"""
    probs = torch.bernoulli(losses / losses.norm())
    loss = torch.sum(losses * probs) / probs.sum()
    loss.backward()