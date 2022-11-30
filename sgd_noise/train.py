from .wgrad import weighted_backward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Callable, Union, List, Tuple


def train_one_epoch(model: nn.Module, opt: optim.Optimizer, criterion: nn.Module, dl: DataLoader, wgrad: bool = True) \
        -> float:
    model_device = next(model.parameters()).device
    runnning_loss = 0
    model.train()
    for x, y in dl:
        opt.zero_grad()
        output = model(x.to(model_device))
        losses = criterion(output, y.to(model_device))
        if wgrad:
            weighted_backward(model, losses)
            runnning_loss += losses.mean().item()
        else:
            loss = losses.mean()
            loss.backward()
            runnning_loss += loss.item()
        opt.step()
    return runnning_loss / len(dl)


def validate_model(model: nn.Module, dl: DataLoader, metrics: Union[List[Callable], Callable],
                   save_preds: bool = False) -> Union[Tuple[np.ndarray, pd.DataFrame], Tuple[np.ndarray, None]]:
    model_device = next(model.parameters()).device

    if not isinstance(metrics, list):
        metrics = [metrics]
    dfs = []

    model.eval()
    with torch.inference_mode():
        running_loss = np.zeros(shape=len(metrics))
        for x, y in dl:
            preds = model(x.to(model_device))
            for i, metric in enumerate(metrics):
                loss = metric(preds, y.to(model_device))
                running_loss[i] += loss.item()
            if save_preds:
                dfs.append(pd.DataFrame({'pred': preds.cpu().numpy()}))
    if save_preds:
        all_preds = pd.concat(dfs)
        return running_loss / len(dl), all_preds
    else:
        return running_loss / len(dl), None


def train(model: nn.Module, opt: optim.Optimizer, criterion: nn.Module, epochs: int, dl: DataLoader,
          vdl: DataLoader = None, metrics: List[Callable] = [], wgrad: bool = False, eval_train_loss: bool = True,
          print_losses: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if wgrad:
        assert hasattr(model, 'alpha'), 'If wgrad=True, model must be WeightedGradient'

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, opt, criterion, dl, wgrad)
        if eval_train_loss:
            train_loss = validate_model(model, dl, metrics)[0][0]
        else:
            train_loss = epoch_loss
        train_losses.append(train_loss)

        if vdl is not None:
            val_loss, _ = validate_model(model, vdl, metrics)
            val_losses.append(val_loss[0])
            model.train()
            if print_losses:
                print('Epoch: ', epoch, '    Train loss: ', round(train_loss, 6),
                      '    Val loss: ', *[round(v, 6) for v in val_loss])

        else:
            if print_losses:
                print('Epoch: ', epoch, '    Train loss: ', round(train_loss, 6))

    return torch.tensor(train_losses), torch.tensor(val_losses)
