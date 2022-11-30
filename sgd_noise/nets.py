import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=nn.ReLU, final_activation=None,
                 batch_norm=True, batch_norm_init=True):
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__()
        self.num_layers = len(layer_sizes)
        layers = []

        if batch_norm_init:
            layers += [nn.BatchNorm1d(input_size)]
        layers += [nn.Linear(input_size, layer_sizes[0])]
        if batch_norm:
            layers += [nn.BatchNorm1d(layer_sizes[0])]
        layers += [activation()]

        for i in range(self.num_layers - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1])]
            if batch_norm:
                layers += [nn.BatchNorm1d(layer_sizes[i + 1])]
            layers += [activation()]
        layers += [nn.Linear(layer_sizes[self.num_layers - 1], output_size)]
        if final_activation is not None:
            layers += [final_activation()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
