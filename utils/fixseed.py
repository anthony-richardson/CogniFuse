import numpy as np
import torch
import random


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Required for true reproducibility. Forces determinism at the cost of slightly slower training.
    # NOTE: CUBLAS_WORKSPACE_CONFIG=:4096:8 needs to be placed before script execution.
    torch.use_deterministic_algorithms(True)
