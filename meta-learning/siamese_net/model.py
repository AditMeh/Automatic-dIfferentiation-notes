import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, x1, x2):

        # image 1
        out_1 = F.max_pool2d(torch.relu(self.conv1(x1)),
                             stride=2, kernel_size=2)
        out_1 = F.max_pool2d(torch.relu(self.conv2(out_1)),
                             stride=2, kernel_size=2)
        out_1 = F.max_pool2d(torch.relu(self.conv3(out_1)),
                             stride=2, kernel_size=2)
        out_1 = torch.relu(self.conv4(out_1))

        # image 2
        out_2 = F.max_pool2d(torch.relu(self.conv1(x2)),
                             stride=2, kernel_size=2)
        out_2 = F.max_pool2d(torch.relu(self.conv2(out_2)),
                             stride=2, kernel_size=2)
        out_2 = F.max_pool2d(torch.relu(self.conv3(out_2)),
                             stride=2, kernel_size=2)
        out_2 = torch.relu(self.conv4(out_2))

        out_1, out_2 = self.fc1(
            out_1.view(-1, 256 * 6 * 6)), self.fc1(out_2.view(-1, 256 * 6 * 6))

        pred = torch.sigmoid(self.fc2(torch.abs(out_1 - out_2)))

        return pred
