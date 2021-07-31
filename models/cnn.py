import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class EncoderCA(nn.Module):
    def __init__(self, ndims: int = 1):
        super(EncoderCA, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, padding=0)
        self.conv2 = nn.Conv2d(5, 15, 5, padding=0)
        self.conv3 = nn.Conv2d(15, 25, 5, padding=0)
        self.conv4 = nn.Conv2d(25, ndims, 5, padding=0)
        self.rec_field = 5 + 4 * 3

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.relu(self.conv4(x))


class DecoderCA(nn.Module):
    def __init__(self, ndims: int = 1):
        super(DecoderCA, self).__init__()
        self.conv1 = nn.Conv2d(ndims, 5, 5, padding=0)
        self.conv2 = nn.Conv2d(5, 15, 5, padding=0)
        self.conv3 = nn.Conv2d(15, 25, 5, padding=0)
        self.conv4 = nn.Conv2d(25, 1, 5, padding=0)
        self.rec_field = 5 + 4 * 3

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.relu(self.conv4(x))
    

class EncoderCA_3D(nn.Module):
    def __init__(self, num_out_channels: int = 1):
        super(EncoderCA_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, 3, padding=0)
        self.conv2 = nn.Conv3d(5, 15, 3, padding=0)
        self.conv3 = nn.Conv3d(15, 25, 3, padding=0)
        self.conv4 = nn.Conv3d(25, num_out_channels, 3, padding=0)
        self.rec_field = 3 + 2 * 3

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))


class DecoderCA_3D(nn.Module):
    def __init__(self):
        super(DecoderCA_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, 3, padding=0)
        self.conv2 = nn.Conv3d(5, 15, 3, padding=0)
        self.conv3 = nn.Conv3d(15, 25, 3, padding=0)
        self.conv4 = nn.Conv3d(25, 1, 3, padding=0)
        self.rec_field = 3 + 2 * 3

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))
    

class EncoderCA_3D2(nn.Module):
    def __init__(self):
        super(EncoderCA_3D2, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, 3, padding=1)
        self.conv2 = nn.Conv3d(5, 15, 3, padding=1)
        self.conv3 = nn.Conv3d(15, 25, 3, padding=1)
        self.conv4 = nn.Conv3d(25, 1, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))


class DecoderCA_3D2(nn.Module):
    def __init__(self):
        super(DecoderCA_3D2, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, 3, padding=1)
        self.conv2 = nn.Conv3d(5, 15, 3, padding=1)
        self.conv3 = nn.Conv3d(15, 25, 3, padding=1)
        self.conv4 = nn.Conv3d(25, 1, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))


class Encoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 5, 5, padding=2)
        self.conv2 = nn.Conv2d(5, 15, 5, padding=2)
        self.conv3 = nn.Conv2d(15, 25, 5, padding=2)
        self.conv4 = nn.Conv2d(25, num_channels, 5, padding=2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return torch.tanh(self.conv4(x))


class Decoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 5, 5, padding=2)
        self.conv2 = nn.Conv2d(5, 15, 5, padding=2)
        self.conv3 = nn.Conv2d(15, 25, 5, padding=2)
        self.conv4 = nn.Conv2d(25, num_channels, 5, padding=2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return torch.sigmoid(self.conv4(x))

    
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.MaxPool2d(2),
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
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class TTClassifier(nn.Module):
    '''
    init_cores: batch_size x -1, needed only for initialization of size
    '''
    def __init__(self, init_cores, num_classes, device='cuda'):
        super(TTClassifier, self).__init__()
                
        self.inpts = [None] * len(init_cores)        
        lens = [None] * len(init_cores)
        for i, core in enumerate(init_cores):
            lens[i] = np.prod(core.shape[1:])
            self.inpts[i] = nn.Sequential(
                nn.Linear(lens[i], 2 * lens[i]),
                nn.BatchNorm1d(2 * lens[i]),
                nn.ReLU()
            ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * np.sum(lens), 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        inpts = [None] * len(x)
        for i, core in enumerate(x):
            inpts[i] = self.inpts[i](core)

        return self.classifier(torch.cat(inpts, dim=1))
    
