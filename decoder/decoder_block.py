import torch.nn as nn
import torch.nn.functional as F
from layer.Residual_connection_layer import ResidualConnectionLayer

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]
    
    def forward(self, tgt_sentence, context_vector, tgt_mask, src_tgt_mask):
        out = tgt_sentence
        out = self.residuals[0](out, lambda out : self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out : self.cross_attention(query=out, key=context_vector, value=context_vector, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out
