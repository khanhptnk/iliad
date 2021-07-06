import os
import sys
import math

import torch
import torch.nn as nn


from .base import *
from .transformer_base import *


class TransformerPathEncoder(nn.Module):

    def __init__(self, config):

        super(TransformerPathEncoder, self).__init__()

        self.drop = nn.Dropout(p=config.dropout_ratio)

        visual_feature_with_loc_size = \
            config.img_feature_size + config.loc_embed_size
        input_size = visual_feature_with_loc_size + config.img_feature_size

        self.encoder = TransformerDecoder(
            SelfAttentionLayer,
            input_size,
            config.hidden_size,
            config.attention_heads,
            config.num_layers,
            config.dropout_ratio,
            config.device
        )

        self.curr_view_attention = Attention(
            config.hidden_size, config.img_feature_size, config.hidden_size)

        self.time_encoding = TimeEncoding(config.hidden_size)

        self.device = config.device

    def forward(self, action_embed_seqs, view_feature_seqs):

        batch_size = action_embed_seqs[0].shape[0]

        h_seqs = self.encoder.init_state(batch_size)

        time = self.time_encoding.init(batch_size)

        zipped_info = zip(action_embed_seqs, view_feature_seqs)
        context = []
        for action_embed, view_features in zipped_info:
            attended_view = self.curr_view_attention(
                h_seqs[-1][-1], view_features)
            input = torch.cat([action_embed, attended_view], dim=1)
            output, new_h = self.encoder(input, time, h_seqs)

            context.append(output.unsqueeze(0))

            for h_seq, h in zip(h_seqs, new_h):
                h_seq.append(h)

            time = self.time_encoding(time)

        context = torch.cat(context, dim=0).transpose(0, 1).contiguous()

        return context, new_h


class DescriberTransformerSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(DescriberTransformerSeq2SeqModel, self).__init__()

        vocab_size = config.vocab_size
        pad_idx = config.pad_idx
        hidden_size = config.hidden_size
        dropout = config.dropout_ratio
        num_layers = config.num_layers
        attention_heads = config.attention_heads
        img_feature_size = config.img_feature_size
        word_embed_size = config.word_embed_size
        loc_embed_size = config.loc_embed_size
        device = config.device

        self.instr_embedding = nn.Embedding(vocab_size, word_embed_size,
            padding_idx=pad_idx)

        self.encoder = TransformerPathEncoder(config)

        self.dec_time_encoding = TimeEncoding(hidden_size)

        self.decoder = TransformerDecoder(
            SelfAndSourceAttentionLayer,
            word_embed_size,
            hidden_size,
            attention_heads,
            num_layers,
            dropout,
            device)

        self.predictor = nn.Linear(config.hidden_size, config.n_actions)

        self.enc2dec = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )

        self.device = config.device

    def init_encoder(self, batch_size):
        self.encoder.init(batch_size)

    def encode(self, action_embed_seqs, view_feature_seqs):

        batch_size = action_embed_seqs[0].shape[0]

        context, last_h = self.encoder(action_embed_seqs, view_feature_seqs)
        dec_h_seqs = [[self.enc2dec(h)] for h in last_h]
        dec_time = self.dec_time_encoding.init(batch_size)

        return dec_h_seqs, dec_time, context

    def decode(self, dec_h_seqs, dec_time, prev_action, path_encodings, path_masks):

        prev_action = self.instr_embedding(prev_action)

        output, new_dec_h = self.decoder(prev_action, dec_time, dec_h_seqs,
            path_encodings, path_masks)

        for h_seq, h in zip(dec_h_seqs, new_dec_h):
            h_seq.append(h)

        logit = self.predictor(output)

        new_dec_time = self.dec_time_encoding(dec_time)

        return dec_h_seqs, new_dec_time, logit
