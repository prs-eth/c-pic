import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from typing import Iterable

class Encoder(nn.Module):
    def __init__(self, num_neurons: Iterable[int], num_dims: int):
        super(Encoder, self).__init__()
        self.num_dims = num_dims
        self.rec_field = int(np.ceil((num_neurons[0] ** (1 / num_dims))))
        assert np.allclose(self.rec_field, num_neurons[0] ** (1 / num_dims))
        self.layers = []
        for i, n in enumerate(num_neurons[:-2]):
            self.layers.append(nn.Linear(n, num_neurons[i + 1]))
            self.layers.append(nn.BatchNorm1d(num_neurons[i + 1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(num_neurons[-2], num_neurons[-1]))
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.reshape(-1, self.rec_field ** self.num_dims)
        return self.nn(x)


class Decoder(nn.Module):
    def __init__(self, num_neurons, num_dims):
        super(Decoder, self).__init__()
        self.num_dims = num_dims
        self.rec_field = int(np.ceil((num_neurons[0] ** (1 / num_dims))))
        assert np.allclose(self.rec_field, num_neurons[0] ** (1 / num_dims))
        self.layers = []
        for i, n in enumerate(num_neurons[:-2]):
            self.layers.append(nn.Linear(n, num_neurons[i + 1]))
            self.layers.append(nn.BatchNorm1d(num_neurons[i + 1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(num_neurons[-2], num_neurons[-1]))
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.reshape(-1, self.rec_field ** self.num_dims)
        return self.nn(x)


class Classifier(nn.Module):
    def __init__(self, num_input_features, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_input_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class Regressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.regressor(x)


class TwoHeadsRegressor(nn.Module):
    def __init__(
            self,
            num_input_features1: int,
            num_input_features2: int,
            num_out_features: int):
        super(TwoHeadsRegressor, self).__init__()
        self.num_input_features1 = num_input_features1
        print('Number of input features: {}'.format(self.num_input_features1))

        if num_input_features1 > 0:
            self.mlp1 = nn.Sequential(
                nn.Linear(num_input_features1, 500),
                # nn.BatchNorm1d(500),
                nn.ReLU(),
                nn.Linear(500, 100),
                # nn.BatchNorm1d(100),
                nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Linear(num_input_features2, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, num_out_features))

    def forward(self, x1, x2):
        if self.num_input_features1 > 0:
            x = self.mlp1(x2[:, :self.num_input_features1])
            x = x + torch.normal(0, 0.2, x.shape).to(x.device)
            x1 = torch.cat([x1, x], axis=1)
        return self.mlp2(x1)


class TwoHeadsRegressorNoBN(nn.Module):
    def __init__(
            self,
            num_input_features1: int,
            num_input_features2: int,
            num_out_features: int):
        super(TwoHeadsRegressor, self).__init__()
        self.num_input_features1 = num_input_features1
        print('Number of input features: {}'.format(self.num_input_features1))

        if num_input_features1 > 0:
            self.mlp1 = nn.Sequential(
                nn.Linear(num_input_features1, 500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Linear(num_input_features2, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, num_out_features))


    def forward(self, x1, x2):
        if self.num_input_features1 > 0:
            x = self.mlp1(x2[:, :self.num_input_features1])
            x = x + torch.normal(0, 0.2, x.shape).to(x.device)
            x1 = torch.cat([x1, x], axis=1)
        return self.mlp2(x1)
