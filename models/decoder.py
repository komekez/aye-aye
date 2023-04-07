import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, nz, output_size):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(nz, 100),
            nn.ReLU(),
            nn.Linear(100, 10000),
            nn.ReLU(),
            nn.Linear(10000, self.output_size)
        )

    def forward(self, z):
        return self.net(z)