import torch
import torch.nn as nn
import math

import numpy as np
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_perc):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_perc)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.dropout(self.embed(x)) * math.sqrt(float(self.d_model))


class StaticExpansionBlock(nn.Module):
    def __init__(self, d_model, num_enc_exp_list, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model
        self.num_enc_exp_list = num_enc_exp_list

        self.query_exp_vectors = nn.Embedding(sum(num_enc_exp_list), d_model)
        self.bias_exp_vectors = nn.Embedding(sum(num_enc_exp_list), d_model)

        self.key_embed = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_a_embed1 = nn.Linear(d_model, d_model)

        self.class_b_embed = nn.Linear(d_model, d_model)
        self.class_b_embed1 = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)

        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, mask):
        bs, enc_len, _ = x.shape

        query_exp = self.query_exp_vectors(n_indexes)
        bias_exp = self.bias_exp_vectors(n_indexes)
        x_key = self.key_embed(x)

        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / (self.d_model ** 0.5)
        z = self.Z_dropout(z)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mask == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mask == 0, 0.0)
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_a_embed = self.class_a_embed(x)
        class_a_embed_attn = self.class_a_embed1(class_a_embed)
        class_a_embed = class_a_embed * torch.sigmoid(class_a_embed_attn)
        class_b_embed = self.class_b_embed(x)
        class_b_embed_attn = self.class_b_embed1(class_b_embed)
        class_b_embed = class_b_embed * torch.sigmoid(class_b_embed_attn)
        class_a = torch.matmul(class_a_fw, class_a_embed) + bias_exp
        class_b = torch.matmul(class_b_fw, class_b_embed) + bias_exp
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))

        accum = 0
        class_a_bw_list = []
        class_b_bw_list = []
        for j in range(len(self.num_enc_exp_list)):
            from_idx = accum
            to_idx = accum + self.num_enc_exp_list[j]
            accum += self.num_enc_exp_list[j]
            class_a_bw_list.append(class_a_bw[:, :, from_idx:to_idx] / (class_a_bw[:, :, from_idx:to_idx].sum(dim=-1, keepdim=True) + self.eps))
            class_b_bw_list.append(class_b_bw[:, :, from_idx:to_idx] / (class_b_bw[:, :, from_idx:to_idx].sum(dim=-1, keepdim=True) + self.eps))
        class_a_bw = torch.cat(class_a_bw_list, dim=-1)
        class_b_bw = torch.cat(class_b_bw_list, dim=-1)

        class_a = torch.matmul(class_a_bw, class_a) / len(self.num_enc_exp_list)
        class_b = torch.matmul(class_b_bw, class_b) / len(self.num_enc_exp_list)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_enc_exp_list, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha = MultiHeadAttentionV2(d_model, num_heads, dropout_perc, attention_module=ScaledDotProductAttentionMemory)

        self.stc_exp = StaticExpansionBlock(d_model, num_enc_exp_list, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, mask, attn_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.stc_exp(x=x2, n_indexes=n_indexes, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha(x2, x2, x2, attn_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class DynamicExpansionBlock(nn.Module):
    def __init__(self, d_model, num_exp, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model

        self.num_exp = num_exp
        self.cond_embed = nn.Linear(d_model, d_model)

        self.query_exp_vectors = nn.Embedding(self.num_exp, d_model)
        self.bias_exp_vectors = nn.Embedding(self.num_exp, d_model)

        self.key_linear = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_a_embed1 = nn.Linear(d_model, d_model)

        self.class_b_embed = nn.Linear(d_model, d_model)
        self.class_b_embed1 = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)
        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, mask):
        bs, dec_len, _ = x.shape

        cond = self.cond_embed(x).view(bs, dec_len, 1, self.d_model)
        query_exp = self.query_exp_vectors(n_indexes).unsqueeze(1)
        bias_exp = self.bias_exp_vectors(n_indexes).unsqueeze(1)
        query_exp = (query_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)
        bias_exp = (bias_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)

        x_key = self.key_linear(x)
        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / (self.d_model ** 0.5)
        z = self.Z_dropout(z)

        mod_mask_1 = mask.unsqueeze(2).expand(bs, dec_len, self.num_exp, dec_len).contiguous(). \
            view(bs, dec_len * self.num_exp, dec_len)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_a_embed = self.class_a_embed(x)
        class_a_embed_attn = self.class_a_embed1(class_a_embed)
        class_a_embed = class_a_embed * torch.sigmoid(class_a_embed_attn)
        class_b_embed = self.class_b_embed(x)
        class_b_embed_attn = self.class_b_embed1(class_b_embed)
        class_b_embed = class_b_embed * torch.sigmoid(class_b_embed_attn)
        class_a = torch.matmul(class_a_fw, class_a_embed) + bias_exp
        class_b = torch.matmul(class_b_fw, class_b_embed) + bias_exp
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        mod_mask_2 = mask.unsqueeze(-1).expand(bs, dec_len, dec_len, self.num_exp).contiguous(). \
            view(bs, dec_len, dec_len * self.num_exp)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))
        class_a_bw = class_a_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_b_bw = class_b_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_a_bw = class_a_bw / (class_a_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_bw = class_b_bw / (class_b_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_bw, class_a + bias_exp)
        class_b = torch.matmul(class_b_bw, class_b + bias_exp)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_exp, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha = MultiHeadAttentionV2(d_model, num_heads, dropout_perc, attention_module=ScaledDotProductAttentionMemory)
        self.dyn_exp = DynamicExpansionBlock(d_model, num_exp, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, cross_connection_x, input_attention_mask, cross_attention_mask):

        # Pre-LayerNorm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.dyn_exp(x=x2, n_indexes=n_indexes, mask=input_attention_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha(x2, cross_connection_x, cross_connection_x,
                                        cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_perc):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "num heads must be multiple of d_model"

        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, self.d_k * num_heads)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads)
        self.Wv = nn.Linear(d_model, self.d_k * num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size, q_seq_len, _ = q.shape
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        k_proj = self.Wk(k).view(batch_size, k_seq_len, self.num_heads, self.d_k)
        q_proj = self.Wq(q).view(batch_size, q_seq_len, self.num_heads, self.d_k)
        v_proj = self.Wv(v).view(batch_size, v_seq_len, self.num_heads, self.d_k)

        k_proj = k_proj.transpose(2, 1)
        q_proj = q_proj.transpose(2, 1)
        v_proj = v_proj.transpose(2, 1)

        sim_scores = torch.matmul(q_proj, k_proj.transpose(3, 2))
        sim_scores = sim_scores / self.d_k ** 0.5

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            sim_scores = sim_scores.masked_fill(mask == 0, value=-1e4)
        sim_scores = F.softmax(input=sim_scores, dim=-1)

        attention_applied = torch.matmul(sim_scores, v_proj)
        attention_applied_concatenated = attention_applied.permute(0, 2, 1, 3).contiguous()\
            .view(batch_size, q_seq_len, self.d_model)

        out = self.out_linear(attention_applied_concatenated)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_perc)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout_perc)


    def forward(self, x):
        x2 = self.dropout_1(F.relu(self.linear_1(x)))
        x2 = self.dropout_1(F.relu(self.linear_2(x2)))
        return x + x2

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class ScaledDotProductAttentionMemory(nn.Module):
    '''
    Scaled dot-product attention with memory
    '''

    def __init__(self, d_model, d_k, d_v, h, m=40):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(ScaledDotProductAttentionMemory, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
        if attention_mask is not None:
            # print(att[:, :, :, :nk].shape)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.h, 1, 1)
            # print(attention_mask.shape)

            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask == 0, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttentionV2(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttentionV2, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        d_k = int(d_model / h)
        d_v = d_k
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
        return out
