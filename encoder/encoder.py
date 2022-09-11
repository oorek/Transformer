import torch.nn as nn
import torch.nn.functional as F
import copy

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out