import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, src, src_mask):
        #out = self.encoder(src, src_mask)
        out = self.encoder(self.src_embed(src), src_mask)
        return out

    def decode(self, tgt_sentence, context, tgt_mask, src_tgt_mask):
        #out = self.decoder(tgt_sentence,context, tgt_mask, src_tgt_mask)
        out = self.decoder(self.tgt_embed(tgt_sentence), context, tgt_mask, src_tgt_mask)
        return out
    
    def forward(self, src_sentence, tgt_sentence):
        src_mask = self.make_src_mask(src_sentence)
        tgt_mask = self.make_tgt_mask(tgt_sentence)
        src_tgt_mask = self.make_src_tgt_mask(tgt_sentence, src_sentence)
        context_vector = self.encode(src_sentence, src_mask)
        out = self.decode(tgt_sentence, context_vector, tgt_mask, src_tgt_mask)
        return out
    
    def make_pad_mask(self, query, key, pad_idx=1): #pad_idx : pad 해야하는 index들 탐색
        # query : ( n_batch, query_seq_len)
        # key : (n_batch, key_seq_len)
        query_seq_len , key_seq_len = query.size(1), key.size(1)
        
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) #1이 아닌것들 1, 1이면 0 #(n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1,1,query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len) #(n_batch, 1, query_seq_len, key_seq_len)
        
        mask = key_mask & query_mask
        mask.requires_grad = False #??
        return mask
        
        # 왜 query, key 두개가 필요한거임;;
        # query, key가 각각 무엇을 의미하는가

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
    
    def make_subsequent_mask(self, query, key):
        # query : ( n_batch, query_seq_len)
        # key : (n_batch, key_seq_len) ????
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        # 이 두줄이 이해가 안감 왜 있음?
        #걍 numpy 배열 리턴하면 안됨?
        return mask
    
    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask
    
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask
