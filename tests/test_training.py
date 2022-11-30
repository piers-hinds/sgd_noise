from .data import *
from .models import *


def test_train_one_epoch(linear_dataloader, linear_model):
    dl = linear_dataloader(10, 1)
    lm = linear_model(1)
    opt = torch.optim.SGD(lm.parameters(), lr=0.01)
    loss = train_one_epoch(lm, opt, nn.MSELoss(reduction='none'), dl)
    assert isinstance(loss, float)


def test_validate_model(linear_dataloader, linear_model):
    dl = linear_dataloader(10, 1)
    lm = linear_model(1)
    losses, _ = validate_model(lm, dl, nn.MSELoss())
    losses, _ = validate_model(lm, dl, [nn.MSELoss(), nn.L1Loss()])
    assert isinstance(losses, np.ndarray)


def test_train(linear_dataloader, linear_model):
    dl = linear_dataloader(10, 1)
    lm = linear_model(1)
    opt = torch.optim.SGD(lm.parameters(), lr=0.01)
    out = train(lm, opt, nn.MSELoss(reduction='none'), 2, dl, dl, metrics=[nn.MSELoss(), nn.L1Loss()],
                print_losses=False)


def test_train_wgrad(linear_dataloader, linear_model):
    dl = linear_dataloader(10, 1)
    lm = WeightedGradient(linear_model(1))
    opt = torch.optim.SGD(lm.parameters(), lr=0.01)
    out = train(lm, opt, nn.MSELoss(reduction='none'), 2, dl, dl, metrics=[nn.MSELoss(), nn.L1Loss()],
                print_losses=False, wgrad=True)
