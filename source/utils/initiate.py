import os
import json
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


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return config


def setting_path(config: dict) -> dict:
    """
    { ROOT_DIR } / { DATA NAME } / { TAG }
        - models : trained model path
        - labels : missing / anomaly label path
        - data   : processed data path
        - logs   : log files path
    """

    ROOT_DIR = config["core"]["directory"]
    DATA_NAME = config["core"]["data_name"]
    MODEL_TAG = config["core"]["TAG"]

    output_path = os.path.join(ROOT_DIR, DATA_NAME)
    models_path = os.path.join(output_path, MODEL_TAG)
    os.mkdir(ROOT_DIR) if not os.path.exists(ROOT_DIR) else None
    os.mkdir(output_path) if not os.path.exists(output_path) else None

    config["core"]["path"] = models_path
    config["core"]["models_path"] = os.path.join(models_path, "models")

    if not os.path.exists(models_path):
        os.mkdir(models_path)
        os.mkdir(config["core"]["models_path"])

    return config


def init_device():
    """
    Setting device CUDNN option

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device)
