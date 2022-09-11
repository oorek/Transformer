import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff

    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out
    

def calculate_attention(self, query, key, value, mask):
    # query, key, value : (n_batch, seq_len, d_k)
    # mask : (n_batch, seq_len, seq_len)
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2 ,-1)) # (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)

    out = torch.matmul(attention_prob, value)
    return out