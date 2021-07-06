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

        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))

        weighted_context = torch.bmm(attn3, values).squeeze(1)

        return weighted_context


class EltwiseProdScoring(nn.Module):

    def __init__(self, h_dim, a_dim, dot_dim):

        super(EltwiseProdScoring, self).__init__()

        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        target = self.linear_in_h(h).unsqueeze(1)
        context = self.linear_in_a(all_u_t)
        eltprod = torch.mul(target, context)
        logits = self.linear_out(eltprod).squeeze(2)
        return logits


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


class EncoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio, device):

        super(EncoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = LSTMWrapper(
            input_size,
            hidden_size,
            dropout_ratio,
            device,
            batch_first=True)
        self.device = device

    def forward(self, input, h0=None):
        if h0 is None:
            h0 = self.lstm.init_state(input.size(0))
        context, (last_h, last_c) = self.lstm(input, h0)

        return context, last_h, last_c


class DecoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio, device):

        super(DecoderLSTM, self).__init__()

        self.drop = nn.Dropout(p=dropout_ratio)

        self.lstm = LSTMWrapper(
            input_size,
            hidden_size,
            dropout_ratio,
            device)
        self.device = device

    def forward(self, input, h):
        input_drop = self.drop(input)
        output, new_h = self.lstm(input_drop.unsqueeze(0), h)
        output = self.drop(output.squeeze(0))

        return output, new_h
