import torch
import random
import numpy as np
import pandas as pd


def seeding(seed=31):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.set_printoptions(precision=3, sci_mode=False)
    pd.set_option("mode.chained_assignment", None)
    pd.options.display.float_format = "{:.3f}".format
    np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)


def init_device():
    """
    Setting device CUDNN option

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device)
