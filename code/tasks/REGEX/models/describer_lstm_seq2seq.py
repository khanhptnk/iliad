import os
import sys

import torch
import torch.nn as nn


from .base import LSTMWrapper, Attention


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

    def forward(self, input):

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


class DescriberLSTMSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(DescriberLSTMSeq2SeqModel, self).__init__()

        self.encoder = EncoderLSTM(
            config.src_embed_size,
            config.enc_hidden_size,
            config.dropout_ratio,
            config.device)

        self.decoder = DecoderLSTM(
            config.tgt_embed_size,
            config.dec_hidden_size,
            config.dropout_ratio,
            config.device)

        self.attention = Attention(
            config.hidden_size, config.hidden_size, config.hidden_size // 2)

        self.predictor = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.n_actions)
            )

        self.enc2dec = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )

        self.src_embedding = nn.Embedding(
            config.src_vocab_size, config.src_embed_size, config.src_pad_idx)
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size, config.tgt_embed_size, config.tgt_pad_idx)

        self.device = config.device
        self.n_actions = config.n_actions

    def encode(self, src, src_mask=None):

        batch_size = src.shape[0]

        src_input = self.src_embedding(src) # batch x n_ex x len x hidden
        src_input = src_input.mean(dim=1)
        context, last_enc_h, last_enc_c = self.encoder(src_input)
        last_enc_h = self.enc2dec(last_enc_h)

        self.src_context = context
        self.src_mask = src_mask
        self.dec_h = (last_enc_h, last_enc_c)

    def decode(self, prev_action):

        input = self.tgt_embedding(prev_action)
        output, self.dec_h = self.decoder(input, self.dec_h)

        attended_h, _ = self.attention(
            output, self.src_context, mask=self.src_mask)

        feature = torch.cat([output, attended_h], dim=1)
        logit = self.predictor(feature)

        return logit

    def index_select_decoder_state(self, pos):
        self.dec_h = tuple([s.index_select(1, pos) for s in self.dec_h])
        self.dec_t = self.dec_t.index_select(0, pos)



