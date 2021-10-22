import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.color import colorstr
from TIMEBAND.loss import TIMEBANDLoss
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = Logger(__file__)


class TIMEBANDRunner:
    def __init__(
        self,
        config: dict,
        dataset: TIMEBANDDataset,
        models: TIMEBANDModel,
        metric: TIMEBANDMetric,
        losses: TIMEBANDLoss,
        dashboard: TIMEBANDDashboard,
        device: torch.device,
    ) -> None:
        # Set device
        self.device = device

        self.dataset = dataset
        self.models = models
        self.metric = metric
        self.losses = losses
        self.dashboard = dashboard

        # Set Config
        config = self.set_config(config=config)

        self.forecast_len = self.dataset.forecast_len

    def set_config(self, config: dict = None) -> dict:
        """
        Configure settings related to the data set.

        params:
            config: Trainer configuration dict
                `config['trainer']`
        """

        self.print_cfg = print_cfg = config["print"]

        # Train option
        self.lr_config = config["learning_rate"]
        self.lr = config["learning_rate"]["base"]
        self.lr_gammaG = config["learning_rate"]["gammaG"]
        self.lr_gammaD = config["learning_rate"]["gammaD"]

        self.trainer_config = config["epochs"]
        self.epochs = config["epochs"]["iter"]
        self.base_epochs = config["epochs"]["base"]
        self.iter_epochs = config["epochs"]["iter"]
        self.iter_critic = config["epochs"]["critic"]

        self.amplifier = config["amplifier"]

        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_interval = print_cfg["interval"]

        # Visual option
        self.print_cfg = config["print"]

    def run(self, trainset: DataLoader, validset: DataLoader) -> None:
        logger.info("Train the model")

        # Prediction
        self.data_idx = 0
        self.pred_initate()

        # Dashboard
        self.dashboard.init_figure()

        # Train Section
        self.step(trainset)
        self.step(validset)

        # Dashboard
        self.dashboard.clear_figure()

        self.models.load("BEST")
        self.models.save(best=True)

    def train_step(self, dataset):
        def generate(x):
            return self.models.netG(x)[:, : self.forecast_len].to(self.device)

        amplifier = self.amplifier

        losses = self.losses.init_loss()
        score = self.metric.init_score()
        for i, data in enumerate(tqdm(dataset)):
            self.optimD.zero_grad()
            self.optimG.zero_grad()

            true_x = data["encoded"].to(self.device)
            true_y = data["decoded"].to(self.device)
            (batchs, forecast_len, target_dims) = true_y.shape

            fake_y = generate(true_x)
            pred_y = self.dataset.denormalize(fake_y.cpu())

            self.predicts(pred_y)
            preds = torch.tensor(self.preds)

            reals = self.dataset.forecast[
                self.data_idx : self.data_idx + preds.shape[0]
            ]
            self.data_idx += batchs

            self.metric.NMAE(reals, preds.clone().detach())
            self.metric.SCORE(reals, preds * self.amplifier)
            self.metric.RMSE(reals, preds * self.amplifier)

            self.dashboard.visualize(batchs, reals, self.preds * self.amplifier, self.stds)

        return self.metric.nmae / (i + 1)
        
    def pred_initate(self):
        decoded_shape = self.dataset.decode_shape
        (batch_size, forecast_len, target_dims) = decoded_shape

        init_shape3 = (forecast_len - 1, forecast_len, target_dims)
        init_shape2 = (forecast_len - 1, target_dims)

        # version v2.2
        self.pred_data = np.empty(init_shape3)
        self.pred_data[:] = np.nan

        self.preds = np.empty(init_shape2)
        self.preds[:] = np.nan

        self.stds = np.zeros(init_shape2)

    def predicts(self, pred):
        (batch_size, forecast_len, target_dims) = pred.shape
        pred = pred.detach().numpy()

        nan_shape3 = np.empty((batch_size, forecast_len, target_dims))
        nan_shape3[:] = np.nan
        nan_shape2 = np.empty((batch_size, target_dims))
        nan_shape2[:] = np.nan

        self.pred_data = np.concatenate(
            [self.pred_data[1 - forecast_len :], nan_shape3]
        )
        for f in range(forecast_len):
            self.pred_data[f : batch_size + f, f] = pred[:, f]

        self.preds = np.nanmedian(self.pred_data, axis=1)
        self.stds = np.nanstd(self.pred_data, axis=1)

        for f in range(forecast_len - 1, 0, -1):
            gamma = (forecast_len - f) / (forecast_len - 1)
            self.stds[-f] += self.stds[-f - 1] * gamma

        for f in range(1, forecast_len):
            self.stds[f] += self.stds[f - 1] * 0.1
                    
def desc(training, epoch, score, losses):
    process = "Train" if training else "Valid"

    if not training:
        score["SCORE"] = colorstr("bright_red", score["SCORE"]) 
        score["RMSE"] = colorstr("bright_blue", score["RMSE"]) 
        score["NMAE"] = colorstr("bright_red", score["NMAE"])  
        losses["L1"] = colorstr("bright_blue", losses["L1"])   
        losses["L2"] = colorstr("bright_blue", losses["L2"])   
        losses["GP"] = colorstr("bright_blue", losses["GP"])   

    return (
        f"[{process} e{epoch + 1:4d}] "
        f"Score {score['SCORE']} ( NME {score['NME']} / NMAE {score['NMAE']} / RMSE {score['RMSE']} ) "
        f"D {losses['D']} ( R {losses['R']} F {losses['F']} ) "
        f"G {losses['G']} ( G {losses['G_']} L1 {losses['L1']} L2 {losses['L2']} GP {losses['GP']} )"
    )
