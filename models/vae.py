import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, nz, beta=1.0):
        super().__init__()
        self.beta = beta
        self.nz = nz


    def forward(self, x):
        pass


    def loss(self, x, output):
        pass

    def reconstruction(self, x):
        pass