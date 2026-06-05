import torch

def set_seed(seed=0):
    import random, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
