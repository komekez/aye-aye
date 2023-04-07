import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, nz, input_size):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 10000),
            nn.ReLU(),
            nn.Linear(10000, 100),
            nn.ReLU(),
            nn.Linear(100, nz)
        )

    def forward(self, x):
        return self.net(x)