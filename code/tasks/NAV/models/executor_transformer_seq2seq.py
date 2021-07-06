import os
import sys
import math

import torch
import torch.nn as nn


from .base import *
from .transformer_base import *


class ExecutorTransformerSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(ExecutorTransformerSeq2SeqModel, self).__init__()

        self.instr_embedding = nn.Embedding(config.vocab_size,
            config.word_embed_size, padding_idx=config.pad_idx)

        hidden_size = config.hidden_size
        dropout = config.dropout_ratio
        num_layers = config.num_layers
        attention_heads = config.attention_heads
        img_feature_size = config.img_feature_size
        max_instruction_length = config.max_instruction_length
        word_embed_size = config.word_embed_size
        loc_embed_size = config.loc_embed_size
        device = config.device

        self.positional_encoding = PositionalEncoding(
            hidden_size, max_instruction_length + 10)

        self.encoder = TransformerEncoder(
            word_embed_size,
            hidden_size,
            attention_heads,
            num_layers,
            dropout)

        visual_feature_with_loc_size = img_feature_size + loc_embed_size

        text_dec_input_size = visual_feature_with_loc_size + img_feature_size

        self.dec_time_encoding = TimeEncoding(hidden_size)

        self.text_decoder = TransformerDecoder(
            SelfAndSourceAttentionLayer,
            text_dec_input_size,
            hidden_size,
            attention_heads,
            num_layers,
            dropout,
            device)

        state_dec_input_size = hidden_size + img_feature_size

        self.state_decoder = TransformerDecoder(
            SelfAndSimAttentionLayer,
            state_dec_input_size,
            hidden_size,
            attention_heads,
            num_layers,
            dropout,
            device)

        self.curr_visual_attention = Attention(hidden_size,
            img_feature_size, hidden_size)

        self.next_visual_attention = Attention(hidden_size,
            img_feature_size, hidden_size)

        self.predictor = EltwiseProdScoring(
            hidden_size,
            visual_feature_with_loc_size,
            hidden_size)

        self.enc2dec = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )

        self.start_action = torch.zeros(visual_feature_with_loc_size,
            dtype=torch.float, device=config.device)

        self.device = device

    def init_action(self, batch_size):
        return self.start_action.expand(batch_size, -1)

    def encode(self, instr, instr_mask=None):

        batch_size = instr.shape[0]
        input = self.instr_embedding(instr)
        input = input * math.sqrt(input.shape[-1])
        output = self.positional_encoding(input)

        context, last_enc_h = self.encoder(input, instr_mask)

        text_dec_h_seqs = [[self.enc2dec(h)] for h in last_enc_h]
        state_dec_h_seqs = [[self.enc2dec(h)] for h in last_enc_h]
        dec_time = self.dec_time_encoding.init(batch_size)

        return text_dec_h_seqs, state_dec_h_seqs, dec_time, context

    def decode(self, text_dec_h_seqs, state_dec_h_seqs, dec_time, prev_action,
            action_embeds, instr, instr_mask, curr_view_features, logit_mask):

        attended_view = self.curr_visual_attention(
            state_dec_h_seqs[-1][-1], curr_view_features)

        text_dec_input = torch.cat([prev_action, attended_view], dim=1)

        text_dec_output, new_text_dec_h = self.text_decoder(
            text_dec_input, dec_time, text_dec_h_seqs, instr, instr_mask)
        for dec_h_seq, h in zip(text_dec_h_seqs, new_text_dec_h):
            dec_h_seq.append(h)

        attended_next_view = self.next_visual_attention(
            text_dec_output, curr_view_features)

        state_dec_input = torch.cat([text_dec_output, attended_next_view], dim=1)

        state_dec_output, new_state_dec_h = self.state_decoder(
            state_dec_input, dec_time, state_dec_h_seqs)
        for dec_h_seq, h in zip(state_dec_h_seqs, new_state_dec_h):
            dec_h_seq.append(h)

        feature = state_dec_output

        logit = self.predictor(feature, action_embeds)

        logit.masked_fill_(logit_mask, -float('inf'))

        new_dec_time = self.dec_time_encoding(dec_time)

        return text_dec_h_seqs, state_dec_h_seqs, new_dec_time, logit
