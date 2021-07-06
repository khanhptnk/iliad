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


class ExecutorLSTMSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(ExecutorLSTMSeq2SeqModel, self).__init__()

        self.task_encoder = EncoderLSTM(
            config.task_embed_size,
            config.hidden_size,
            config.dropout_ratio,
            config.device)

        self.src_encoder = EncoderLSTM(
            config.src_embed_size,
            config.hidden_size,
            config.dropout_ratio,
            config.device)

        self.decoder = DecoderLSTM(
            config.tgt_embed_size,
            config.hidden_size,
            config.dropout_ratio,
            config.device)

        self.task_attention = Attention(
            config.hidden_size, config.hidden_size, config.hidden_size // 2)
        self.src_attention = Attention(
            config.hidden_size, config.hidden_size, config.hidden_size // 2)

        self.predictor = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.n_actions)
            )

        self.task2tgt = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        self.src2task = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )

        self.task_embedding = nn.Embedding(
            config.task_vocab_size, config.task_embed_size, config.task_pad_idx)
        self.src_embedding = nn.Embedding(
            config.src_vocab_size, config.src_embed_size, config.src_pad_idx)
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size, config.tgt_embed_size, config.tgt_pad_idx)

        self.device = config.device
        self.n_actions = config.n_actions

    def encode_task(self, task, task_mask=None):

        batch_size = task.shape[0]

        task_input = self.task_embedding(task)
        context, last_enc_h, last_enc_c = self.task_encoder(
            task_input, h0=self.last_src_h)
        last_enc_h = self.task2tgt(last_enc_h)

        self.last_task_h = (last_enc_h, last_enc_c)

        self.task_context = context
        self.task_mask = task_mask

    def encode_src(self, src, src_mask=None):

        batch_size = src.shape[0]

        src_embed = self.src_embedding(src)
        src_input = src_embed
        context, last_enc_h, last_enc_c = self.src_encoder(src_input)
        last_enc_h = self.src2task(last_enc_h)

        self.last_src_h = (last_enc_h, last_enc_c)

        self.src_context = context
        self.src_mask = src_mask

    def encode(self, task, task_mask, src, src_mask):

        self.encode_src(src, src_mask)
        self.encode_task(task, task_mask)

        self.dec_h = self.last_task_h

    def decode(self, prev_action):

        input = self.tgt_embedding(prev_action)

        output, self.dec_h = self.decoder(input, self.dec_h)

        attended_task_h, _ = self.task_attention(
            output, self.task_context, mask=self.task_mask)

        attended_src_h, _ = self.src_attention(
            output, self.src_context, mask=self.src_mask)

        feature = torch.cat([output, attended_task_h, attended_src_h], dim=1)
        logit = self.predictor(feature)

        return logit


