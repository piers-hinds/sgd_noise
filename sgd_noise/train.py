import torch
import pandas as pd
import numpy as np


def train(model, opt, criterion, epochs, dl, vdl=None, metrics=[], eval_train_loss=True, print_losses=True):
    model.train()
    losses = []
    val_losses = []
    for epoch in range(epochs):
        running_loss = 0
        for x, y in dl:
            opt.zero_grad()
            preds = model(x.to(next(model.parameters()).device))
            loss = criterion(preds, y.to(next(model.parameters()).device))
            loss.backward()
            opt.step()
            running_loss += loss.item()

        if eval_train_loss:
            train_loss = validate_model(model, dl, metrics)[0][0]
        else:
            train_loss = running_loss / len(dl)
        losses.append(train_loss)

        if vdl is not None:
            val_loss, _ = validate_model(model, vdl, metrics)
            model.train()
            if print_losses:
                print('Epoch: ', epoch, '    Train loss: ', round(train_loss, 6),
                      '    Val loss: ', *[round(v, 6) for v in val_loss])

        else:
            if print_losses:
                print('Epoch: ', epoch, '    Train loss: ', round(train_loss, 6))
        val_losses.append(val_loss[0])
    return torch.tensor(losses), torch.tensor(val_losses)


def validate_model(model, dl, metrics, save_preds=False):
    if not isinstance(metrics, list):
        metrics = [metrics]
    dfs = []
    model.eval()
    with torch.inference_mode():
        running_loss = np.zeros(shape=len(metrics))
        for x, y in dl:
            preds = model(x.to(next(model.parameters()).device))
            for i, metric in enumerate(metrics):
                loss = metric(preds, y.to(next(model.parameters()).device))
                running_loss[i] += loss.item()
            if save_preds:
                dfs.append(pd.DataFrame({'pred': preds.cpu().numpy()}))
    if save_preds:
        all_preds = pd.concat(dfs)
        return running_loss / len(dl), all_preds
    else:
        return running_loss / len(dl), None
