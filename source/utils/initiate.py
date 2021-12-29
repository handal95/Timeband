import os
import torch
import random
import numpy as np
import pandas as pd
from typing import List

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
    

def check_dirs_exist(basepath: str, dirlist: list, force=False):
    for dir in dirlist:
        dirpath = os.path.join(basepath, dir)

        if not os.path.exists(dirpath):
            DIR_NOT_FOUND_MESSAGE = f"{dirpath} is Not Exists!"

            if force:
                raise FileExistsError(DIR_NOT_FOUND_MESSAGE)
            else:
                print(DIR_NOT_FOUND_MESSAGE)
        
        os.makedirs(dirpath, exist_ok=True)
        

def check_files_exist(filelist: List[str], force=False):
    basepath = os.path.dirname(os.getcwd())

    for file in filelist:
        filepath = os.path.join(basepath, file)
        
        if not os.path.exists(filepath):
            FILE_NOT_FOUND_MESSAGE = f"{filepath} is Not Exists!"

            if force:
                raise FileExistsError(FILE_NOT_FOUND_MESSAGE)
            else:
                print(FILE_NOT_FOUND_MESSAGE)
        


def init_device():
    """
    Setting device CUDNN option

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device)
