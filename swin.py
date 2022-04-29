import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def window_partition(x, window_size):
    '''
    Args:
        x: b x t x d_model
            - b: batch size
            - t: total input size (#patches)
            - d_model: embedding dimension
        window_size: local window size (int)
    Do:
        b x t x d_model -> n x w x d_model
            - w: local window size
            - n: b * #windows (b * (t/w))
    '''

    assert len(x.shape) == 3
    b, t, d = x.shape
    x = x.view(-1, window_size, d)
    
    return x


def window_reverse(x, batch_size):
    '''
    Args:
        x: n x w x d_model
            - n: b * #windows (b * (t/w))
            - w: local window size
            - d_model: embedding dimension
        window_size: local window size (int)
    Do:
        n x w x d_model -> b x t x d_model
            - b: batch size
            - t: total input size (#patches)
    '''

    assert len(x.shape) == 3
    x = rearrange(x, "(b n) w d -> b (n w) d", b=batch_size)

    return x


class ShotEmbedding(nn.Module): 
    def __init__(self, cfg):
        super().__init__()

        '''
        Args:
            cfg.input_dim: 2048 (frome shot encoder)
            cfg.hidden_size: 768 (embedding dimension -> d_model)
            cfg.hidden_dropout_prob:
        Do: 
            b x 2K x 2048(input_dim) -> b x 2K x d_model
                - b: batch size
                - K: initial neighbor size
                - d_model: embedding dimension
            + positional embedding
        '''

        nn_size = cfg.neighbor_size # 2*K -> model/__init__.py에서 cfg에 neighbor_size 추가해야됨
        self.shot_embedding = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.position_embedding = nn.Embedding(nn_size, cfg.hidden_size)
        self.register_buffer("pos_ids", torch.arange(nn_size, dtype=torch.long))
        # self.mask_embedding = -> 안 필요할듯 MSM에서 사용하는 것 같음

        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x):
        shot_emb = self.shot_embedding(x)
        pos_emb = self.position_embedding(self.pos_ids)
        embeddings = shot_emb + pos_emb
        embeddings = self.dropout(self.LayerNorm(embeddings))

        return embeddings


def create_mask(window_size):
    mask = torch.ones(window_size, window_size) # w x w
    assert window_size % 2 == 0
    displacement = window_size // 2

    # mask out quadrand-1,-3
    mask[:displacement, -displacement:] = 0
    mask[-displacement:, :displacement] = 0

    return mask


def scaled_dot_product(q, k, v, mask=None):
    '''
    Args:
        q: Q @ W_q (n x h x w x d_k)
        k: K @ W_k (n x h x w x d_k)
        v: V @ W_v (n x h x w x d_k)
            n: batch size * #windows
            h: #heads
            w: window size
            d_k: hidden dimension
        mask: window_size x window_size
    Do:
        (Q @ K^T / sqrt(d_K)) @ V 
        n x h x w x d_k -> n x h x w x d_k
    '''

    d_k = q.size()[-1]
    attention = torch.matmul(q, k.transpose(-2, -1)) # n x h x w x w
    attention = attention / math.sqrt(d_k) # n x h x w x w
    if mask is not None:
        attention = attention.masked_fill(mask==0, -9e15)
    attention = F.softmax(attention, dim=-1)
    values = torch.matmul(attention, v) # n x h x w x d_k

    return values


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads,
        batch_size, 
        window_size, 
        shifted
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.window_size = window_size
        self.displacement = window_size // 2
        self.shifted = shifted

        self.w_qkv = nn.Linear(self.d_model, 3*d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        '''
        Args:
            x: b x t x d_model
                - b: batch size
                - t: total input size (#patches)
                - d_model: embedding dimension
        Do:
            multi-head self-attention layer
            b x t x d_model -> b x t x d_model
        '''

        if self.shifted:
            x = torch.roll(x, shifts=-self.displacement, dims=1)

        # b x t x d_model -> n x w x d_model
        x = window_partition(x, self.window_size)

        n, w, d = x.shape
        qkv = self.w_qkv(x) # n x w x 3*d_model

        # separate Q, K, V
        qkv = rearrange(qkv, "n w (h d_k qkv) -> n h w (d_k qkv)", h=self.num_heads, qkv=3) # n x h x w x 3*d_k
        q, k, v = qkv.chunk(3, dim=-1) # n x h x w x d_k

        # calculate values
        # n x h x w x d_k
        if self.shifted:
            values = scaled_dot_product(q, k, v, create_mask(self.window_size))
        else:
            values = scaled_dot_product(q, k, v)

        values = rearrange(values, "n h w d_k -> n w (h d_k)") # n x w x d_model
        values = self.w_o(values) # n x w x d_model

        # n x w x d_model -> b x t x d_model
        x = window_reverse(x, self.batch_size)

        if self.shifted:
            x = torch.roll(x, shifts=self.displacement, dims=1)

        return x


class SwinTransformerBlock(nn.Module):
    ...


class Stage(nn.Module):
    ...


class SwinTransformerCRN(nn.Module):
    ...