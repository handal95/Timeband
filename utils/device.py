import torch


def init_device():
    """
    Setting device CUDNN option

    """
    # TODO : Using parallel GPUs options
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device)
