import torch.nn as nn
import torch.nn.functional as F
from layer.Residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)] #index가 필요없을때 _ 사용

    def forward(self, src, src_mask):
        out = src
        out = self.self_attention(query=out, key=out, value=out, mask=src_mask)
        out = self.position_ff(out)
        return out
    

