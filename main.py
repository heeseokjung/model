import torch
from swin import ShotEmbedding
from einops import rearrange
import numpy as np

def window_partition(x, window_size):
    '''
    Args:
        x: (b x (2K) x d_model)
        window_size: local window size (int)
    Do:
        x -> (b*(2K/window_size) x window_size x d_model)
    '''

    assert len(x.shape) == 3
    b, s, d = x.shape # b x (2K) x d_model
    x = x.view(-1, window_size, d)
    
    return x

def window_reverse(x, window_size):
    '''
    Args:
        x: ((b * # windows) x window_size x d_model)
        window_size: local window size (int)
    Do:
        x -> (b x (2k) x d_model)
    '''

    assert len(x.shape) == 3
    x = rearrange(x, "(b n) w d -> b (n w) d", w=window_size)
    return x

def create_mask(window_size):
    mask = torch.ones(window_size, window_size)
    displacement = window_size // 2

    # mask out quadrand-1,-3
    mask[:displacement, -displacement:] = 0
    mask[-displacement:, :displacement] = 0

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


def main():
    '''
    model = ShotEmbedding()
    x = torch.Tensor(4, 16, 2048)
    y = model(x)
    print(y.shape)
    '''

    '''
    x = torch.Tensor(4, 32, 768)
    y = window_partition(x, 4)
    print(y.shape)
    z = window_reverse(y, 4)
    print(z.shape)
    '''

    print(get_relative_distances(4).shape)


if __name__ == "__main__":
    main()