import random
import numpy as np
import torch


def set_seed(seed: float) -> None:
    """
    Deterministic seed config

    :param seed: random seed for the PRNG
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

