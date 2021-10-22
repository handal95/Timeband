import os
import torch
import numpy as np
from utils.logger import Logger
from .utils.lstm_layer import LSTMGenerator as NetG
from .utils.lstm_layer import LSTMDiscriminator as NetD

logger = Logger(__file__)


class TIMEBANDModel:
    """
    TIMEBAND Dataset

    """

    def __init__(self, config: dict, device: torch.device) -> None:
        """
        TIMEBAND Dataset

        Args:
            config: Dataset configuration dict
            device: Torch device (cpu / cuda:0)
        """
        logger.info("  Model: ")

        # Set Config
        self.set_config(config)

        # Set device
        self.device = device

        self.netD = None
        self.netG = None

    def set_config(self, config: dict) -> None:
        """
        Configure settings related to the data set.

        params:
            config: Dataset configuration dict
                `config['models']`
        """
        self.directory = config["directory"]
        self.model_tag = config["model_tag"]
        self.model_dir = os.path.join(self.directory, self.model_tag)
        os.mkdir(self.directory) if not os.path.exists(self.directory) else None
        os.mkdir(self.model_dir) if not os.path.exists(self.model_dir) else None

        self.load_option = config["load"]
        self.save_option = config["save"]
        self.best_score = config["best_score"]

        self.hidden_dim = config["hidden_dim"]
        self.layers_num = config["layers_num"]

    def initiate(self, dims: dict) -> None:
        if self.netD and self.netG:
            return

        enc_dim, dec_dim = dims["encode"], dims["decode"]
        netD = NetD(dec_dim, self.hidden_dim, self.layers_num, self.device)
        netG = NetG(enc_dim, dec_dim, self.hidden_dim, self.layers_num, self.device)

        self.netD, self.netG = netD.to(self.device), netG.to(self.device)
        logger.info(f" - Initiated netD : {self.netD}, netG: {self.netG}")
        self.save()

    def load(self, postfix: str = "") -> tuple((NetD, NetG)):
        netD_path = self.get_path("netD", postfix)
        netG_path = self.get_path("netG", postfix)

        if self.load_option:
            if os.path.exists(netD_path) and os.path.exists(netG_path):
                logger.info(f" - {postfix} Model Loading : {netD_path}, {netG_path}")
                self.netD = torch.load(netD_path)
                self.netG = torch.load(netG_path)
            else:
                logger.warn(f" - {postfix} Model Loading Fail")

        return self.netD, self.netG

    def save(self, postfix: str = "", best: bool = False) -> None:
        netD_path = self.get_path("netD", postfix)
        netG_path = self.get_path("netG", postfix)

        if self.save_option:
            torch.save(self.netD, netD_path)
            torch.save(self.netG, netG_path)

            if best:
                best_netD_path = self.get_path("netD", "BEST")
                best_netG_path = self.get_path("netG", "BEST")
                torch.save(self.netD, best_netD_path)
                torch.save(self.netG, best_netG_path)
                postfix = f"Best({postfix})"

            logger.info(f"*** {postfix} MODEL IS SAVED ***")

    def get_path(self, target: str, postfix: str = "") -> os.path:
        filename = target if postfix == "" else f"{target}_{postfix}"
        filepath = os.path.join(self.model_dir, f"{filename}.pth")
        return filepath
