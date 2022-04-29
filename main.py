import torch
from swin import get_model

class Config():
    def __init__(self):
        self.neighbor_size = 16
        self.input_dim = 2048
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.

def main():
    cfg = Config()
    input = torch.randn(4, 16, 2048)
    model = get_model(cfg=cfg)

    pred = model(input)
    print(pred)

if __name__ == "__main__":
    main()