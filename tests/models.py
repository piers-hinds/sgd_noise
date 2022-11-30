import pytest
import torch
import torch.nn as nn


@pytest.fixture
def linear_model():

    def _method(n: int) -> nn.Module:
        return nn.Linear(n, 1)

    return _method
