import torch
from swin import ShotEmbedding
from einops import rearrange
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    '''
    Args:
        x: b x t x d_model
            - b: batch size
            - t: total input size (#patches)
            - d_model: hidden dimension
        window_size: local window size (int)
    Do:
        b x t x d_model -> b*(t/w) x w x d_model
            - w: local window size
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

def create_mask(window_size):
    mask = torch.ones(window_size, window_size) # w x w
    assert window_size % 2 == 0
    displacement = window_size // 2

    # mask out quadrand-1,-3
    mask[:displacement, -displacement:] = 0
    mask[-displacement:, :displacement] = 0

    return mask


def main():
    x = torch.Tensor([[[[2, 2], [1, 1]]]])
    y = F.softmax(x, dim=-2)
    print(x.shape)
    print(x)
    print(y)


if __name__ == "__main__":
    main()