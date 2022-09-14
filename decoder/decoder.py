import torch.nn as nn
import torch.nn.functional as F
import copy

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
       # self.layers = []
       # for i in range(n_layer):
       #     self.layers.append(copy.deepcopy(decoder_block))
        self.n_layer = n_layer   
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
    
    def forward(self, context_vector, tgt_sentence, tgt_mask, src_tgt_mask):
        #src_tgt_mask는 self-multi-head attention layer에서 넘어온 query, Encoder에서 넘어온 key,value 사이의 pad masking 이다 ...??????
        out = tgt_sentence
        for layer in self.layers:
            out = layer(out, context_vector, tgt_mask, src_tgt_mask)
        return out