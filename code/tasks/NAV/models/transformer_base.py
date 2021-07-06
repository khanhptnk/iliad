import os
import sys
import math
import copy

import torch
import torch.nn as nn

dc = copy.deepcopy
clone = lambda module, n: [dc(module) for _ in range(n)]


class MultiHeadedAttention(nn.Module):
    '''
        Multi-Head Attention module from "Attention is All You Need" (Vaswani et al., 2017)
        Implementation adapted from OpenNMT-py
            https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py#L11
    '''

    def __init__(self, query_dim, key_dim, value_dim, model_dim, head_count):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_key   = nn.Linear(key_dim, model_dim)
        self.linear_value = nn.Linear(value_dim, model_dim)
        self.linear_query = nn.Linear(query_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None):
        '''
            Compute the context vector and the attention vectors.
            Args:
                key (FloatTensor): set of `key_len`
                    key vectors ``(batch, key_len, dim)``
                value (FloatTensor): set of `key_len`
                    value vectors ``(batch, key_len, dim)``
                query (FloatTensor): set of `query_len`
                    query vectors  ``(batch, query_len, dim)``
                mask: binary mask indicating which keys have
                    non-zero attention ``(batch, query_len, key_len)``
            Returns:
                FloatTensor:
                    * output context vectors ``(batch, query_len, dim)``
        '''

        batch_size = key.shape[0]
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        query_ndim = query.dim()
        if query_ndim == 2:
            query = query.unsqueeze(1)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(1)

        # 1) Project key, value, and query.
        key = shape(self.linear_key(key))
        value = shape(self.linear_value(value))
        query = shape(self.linear_query(query))

        key_len = key.shape[2]
        query_len = query.shape[2]

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        context_original = torch.matmul(attn, value)
        context = unshape(context_original)
        output = self.final_linear(context)

        if query_ndim == 2:
            output = output.squeeze(1)

        return output


class SimAttention(nn.Module):
    '''
        Cosine Similarity attention
    '''

    def __init__(self):

        super(SimAttention, self).__init__()

    def forward(self, query, key, value, mask=None):

        dot_product  = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        norm_query   = torch.norm(query, p=2, dim=1)
        norm_key     = torch.norm(key, p=2, dim=2)
        norm_product = norm_query.unsqueeze(1) * norm_key
        norm_product = norm_product.clamp(min=1e-8)

        attn = dot_product / norm_product

        if mask is not None:
            attn.data.masked_fill_(mask, 0)

        attn[attn < 0.9] = 0
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        weighted_context = torch.bmm(attn.unsqueeze(1), value).squeeze(1)

        return weighted_context


class LayerNormResidual(nn.Module):

    def __init__(self, module, input_size, output_size, dropout):

        super(LayerNormResidual, self).__init__()

        self.module = module
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

        if input_size != output_size:
            self.shortcut_layer = nn.Linear(input_size, output_size)
        else:
            self.shortcut_layer = lambda x: x

    def forward(self, input, *args, **kwargs):
        input_shortcut = self.shortcut_layer(input)
        return self.layer_norm(
            input_shortcut + self.dropout(self.module(input, *args, **kwargs)))


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, indices=None):
        if indices is None:
            emb = emb + self.pe[:, :emb.shape[1]]
        else:
            emb = emb + self.pe.squeeze(0).index_select(0, indices)
        return emb


class TimeEncoding(nn.Module):

    def __init__(self, input_size, output_size=None):

        super(TimeEncoding, self).__init__()

        if output_size is None:
            output_size = input_size

        module = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
        self.increment_op = LayerNormResidual(
            module, input_size, output_size, 0)

        init_time = torch.zeros((output_size,), dtype=torch.float)
        self.register_buffer('init_time', init_time)

    def init(self, batch_size):
        return self.init_time.expand(batch_size, -1)

    def forward(self, time):
        return self.increment_op(time)


class TransformerLayer(nn.Module):

    def __init__(self, attention_heads, dropout):

        layer_norm_residual_fn = lambda module, input_size, output_size: \
            LayerNormResidual(module, input_size, output_size, dropout)

        self.attention_fn = lambda query_size, mem_size, output_size: \
            layer_norm_residual_fn(MultiHeadedAttention(
                query_size, mem_size, mem_size, output_size, attention_heads),
                query_size, output_size)

        self.feed_forward_fn = lambda input_size, output_size: \
            layer_norm_residual_fn(
                nn.Sequential(
                    nn.Linear(input_size, output_size * 4),
                    nn.ReLU(),
                    nn.Linear(output_size * 4, output_size)
                ),
                input_size, output_size)

        super(TransformerLayer, self).__init__()


class SelfAttentionLayer(TransformerLayer):

    def __init__(self, query_size, mem_size, output_size, attention_heads, dropout):

        super(SelfAttentionLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(query_size, mem_size, output_size)
        self.feed_forward = self.feed_forward_fn(output_size, output_size)

    def forward(self, input, key, value, mask=None):

        output = self.self_attention(input, key, value, mask)
        output = self.feed_forward(output)

        return output


class SelfAndSourceAttentionLayer(TransformerLayer):

    def __init__(self, query_size, mem_size, output_size, attention_heads, dropout):

        super(SelfAndSourceAttentionLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(
            query_size, mem_size, output_size)
        self.enc_attention  = self.attention_fn(
            output_size, mem_size, output_size)
        self.feed_forward = self.feed_forward_fn(output_size, output_size)

    def forward(self, input, key, value, enc_mem, enc_mask=None):

        hidden = self.self_attention(input, key, value)
        output = self.enc_attention(hidden, enc_mem, enc_mem, enc_mask)
        output = self.feed_forward(output)

        return output


class SelfAndSimAttentionLayer(TransformerLayer):

    def __init__(self, query_size, mem_size, output_size, attention_heads, dropout):

        super(SelfAndSimAttentionLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(
            query_size, mem_size, output_size)

        self.sim_attention = SimAttention()
        self.feed_forward = self.feed_forward_fn(output_size, output_size)
        self.gate = nn.Sequential(
                nn.Linear(output_size * 2, output_size),
                nn.Sigmoid()
            )

    def forward(self, input, key, value):

        self_hidden = self.self_attention(input, key, value)
        self_hidden = self.feed_forward(self_hidden)
        sim_hidden  = self.sim_attention(input, key, value)
        beta = self.gate(torch.cat((self_hidden, sim_hidden), dim=-1))
        output = self_hidden - beta * sim_hidden

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, attention_heads, num_layers, dropout):

        super(TransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        first_layer = SelfAttentionLayer(
            input_size, input_size, hidden_size, attention_heads, dropout)
        hidden_layer = SelfAttentionLayer(
            hidden_size, hidden_size, hidden_size, attention_heads, dropout)

        self.layers = nn.ModuleList(
            [first_layer] + clone(hidden_layer, num_layers - 1))

    def forward(self, input, mask=None):

        input = self.dropout(input)

        last_h = []
        for layer in self.layers:
            output = layer(input, input, input, mask)
            last_h.append(output.mean(dim=1))
            input = output

        return output, last_h


class TransformerDecoder(nn.Module):

    def __init__(self, layer_fn, input_size, hidden_size, attention_heads, num_layers, dropout, device):

        super(TransformerDecoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.input_projection = nn.Linear(input_size, hidden_size)

        hidden_layer = layer_fn(hidden_size, hidden_size, hidden_size,
            attention_heads, dropout)
        self.layers = nn.ModuleList(clone(hidden_layer, num_layers))

        self.init_h = torch.zeros((1, hidden_size)).float().to(device)

    def init_state(self, batch_size):
        h0 = []
        for _ in self.layers:
            h0.append([self.init_h.expand(batch_size, -1).contiguous()])
        return h0

    def forward(self, input, time, h_seqs, *args, **kwargs):

        input = self.input_projection(input) + time
        input = self.dropout(input)

        new_h = []

        for i, layer in enumerate(self.layers):

            key   = torch.stack(h_seqs[i]).transpose(0, 1).contiguous()
            value = torch.stack(h_seqs[i]).transpose(0, 1).contiguous()

            output = layer(input, key, value, *args, **kwargs)
            new_h.append(output)
            input = output

        return output, new_h
