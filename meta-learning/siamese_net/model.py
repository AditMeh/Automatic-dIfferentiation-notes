import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.sister_networks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        out_1 = torch.sigmoid(
            self.fc1(self.sister_networks(x1).view(-1, 256 * 6 * 6)))

        out_2 = torch.sigmoid(
            self.fc1(self.sister_networks(x2).view(-1, 256 * 6 * 6)))

        pred = torch.sigmoid(self.fc2(torch.abs(out_1 - out_2)))

        return pred
