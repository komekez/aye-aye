import torch.nn as nn
from encoder import LinearEncoder
from decoder import LinearDecoder


class VAE(nn.Module):
    def __init__(self, nz, beta=1.0):
        super().__init__()
        self.beta = beta
        self.nz = nz

        self.encoder = LinearEncoder(nz=2*nz, input_size=in_size)
        self.decoder = LinearDecoder(nz=nz, output_size=out_size)
        

    def reparametrization():
        pass

    def forward(self, x):
        pass


    def loss(self, x, output):
        pass

    def reconstruction(self, x):
        pass