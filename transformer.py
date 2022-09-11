import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out

    def decode(self, sentence, context):
        out = self.decoder(sentence,context)
        return out
    
    def forward(self, src, sentence, src_mask):
        encoder_out = self.encode(src, src_mask)
        out = self.decode(sentence, encoder_out)
        return out