from sgd_noise import *
import pytest
import torch
from torch.utils.data import DataLoader


@pytest.fixture
def linear_dataloader():

    def _method(n: int, bs: int) -> DataLoader:
        sigma = 0.05
        true_theta = 0.9

        torch.manual_seed(30)
        noise = torch.randn(n) * sigma
        xs = torch.linspace(0, 1, n)
        ys = true_theta * xs + noise
        dset = TabularDataset(xs.unsqueeze(-1), ys.unsqueeze(-1))
        dl = DataLoader(dset, batch_size=bs)
        return dl

    return _method
