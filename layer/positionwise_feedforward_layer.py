import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU()
        self.fc2 = fc2
        # fc1 : (d_embed, d_ff)
        # fc2 : (d_ff, d_embed)
    def forward(self, x): #(n_batch, seq_len, d_embed)
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out