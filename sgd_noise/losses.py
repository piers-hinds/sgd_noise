import torch
import torch.nn as nn
from abc import abstractmethod, ABC


class _LossWithNoise(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample_noise(self, input: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def elementwise_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        noise = self.sample_noise(input) + 1
        losses = self.elementwise_loss(input, target)
        noisy_loss = (losses * noise).mean()
        return noisy_loss


class _GaussianSampler(_LossWithNoise, ABC):
    def __init__(self, std: float, decay: float = 1) -> None:
        self.std = std
        self.decay = decay
        super().__init__()

    def sample_noise(self, input: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(input) * self.std
        self.std = self.std * self.decay
        return noise


class _MSE:
    def elementwise_loss(self, input, target):
        return (input - target) ** 2


class GaussianMSE(_MSE, _GaussianSampler):
    pass
