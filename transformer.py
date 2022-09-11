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

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)