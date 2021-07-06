import os
import sys

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, h_dim, v_dim, dot_dim):

        super(Attention, self).__init__()

        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, query, values, mask=None):
        target = self.linear_in_h(query).unsqueeze(2)
        context = self.linear_in_v(values)

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))

        weighted_context = torch.bmm(attn3, values).squeeze(1)

        return weighted_context, attn


class LSTMWrapper(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio,
            device, batch_first=False):

        super(LSTMWrapper, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, 1,
            batch_first=batch_first,
            bidirectional=False)

        self.init_h = torch.zeros((1, 1, hidden_size),
            dtype=torch.float, device=device)
        self.device = device

    def init_state(self, batch_size):
        return (self.init_h.expand(-1, batch_size, -1).contiguous(),
                self.init_h.expand(-1, batch_size, -1).contiguous())

    def __call__(self, input, h):
        return self.lstm(input, h)
