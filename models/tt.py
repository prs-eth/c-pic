import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class Encoder(nn.Module):
    def __init__(self, shapes, num_pixels, device='cuda'):
        super(Encoder, self).__init__()
        self.shapes = shapes
        self.device = device
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(5, 5, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(5, 5, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 15, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(15, 15, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(15, 15, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(15, 25, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(25, 25, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(25, 25, 3, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2)
        ).to(device)
        self.cores = [None] * len(shapes)
        
        for i in range(len(shapes)):
            self.cores[i] = nn.Sequential(
                nn.Linear(num_pixels // 2**(2 * 3) * 25, 512),
                nn.Tanh(),
                nn.Linear(512, 1024),
                nn.Tanh(),
                nn.Linear(1024, np.prod(shapes[i])),
                nn.Sigmoid()
            ).to(device)
        
    def forward(self, x):
        x = self.cnn(x.to(self.device))
        x = x.reshape(len(x), -1)
        return [core(x).reshape(-1, *self.shapes[i]) for i, core in enumerate(self.cores)]
