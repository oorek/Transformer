import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_weight, out_weight):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_w = copy.deepcopy(qkv_weight) #(d_embed, d_model)
        self.k_w = copy.deepcopy(qkv_weight)
        self.v_w = copy.deepcopy(qkv_weight)
        self.out_w = out_weight #(d_model, d_embed)
        #deepcopy : 실제로는 다른값들이 들어가기 때문

    def calculate_attention(self, query, key, value, mask):
        # query, key, value : (n_batch, h, seq_len, d_k)
        # mask : (n_batch, 1, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2 ,-1)) # (n_batch, h, seq_len, d_k ) * (n_batch, h, d_k, seq_len) == (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)

        out = torch.matmul(attention_prob, value) #(n_batch, h, seq_len, d_k)
        return out
    
    def forward(self, *args, query, key, value, mask=None):
        # query, key, value = (n_batch, seq_len, d_embed) 초기 임베딩 raw 값
        # mask : (n_batch, seq_len, seq_len)
        # return value : (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc): #(n_batch, seq_len, d_embed)
            out = fc(x) #(n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) #(n_batch, seq_len, h, d_k)
            #view reshape와 비슷
            out = out.transpose(1,2) #(n_batch, h, seq_len, d_k) #calculate attention 함수 형식에 맞게
            return out

        query = transform(query, self.q_w) #(n_batch, h, seq_len, d_k)
        key = transform(key, self.k_w)
        value = transform(value, self.v_w)

        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1,2) #(n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) #(n_batch, seq_len, d_model)
        out = self.out_w(out) #이게 되나?

        return out




            
