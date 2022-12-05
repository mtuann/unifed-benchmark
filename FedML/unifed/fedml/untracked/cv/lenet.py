import torch
from torch import nn
import torch.nn.functional as F

class lenet(nn.Module):

    def __init__(self, output_dim):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, padding_mode = 'reflect')
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x