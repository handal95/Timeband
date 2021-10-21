import torch
import numpy as np

from tqdm import tqdm
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from utils.color import colorstr
from utils.logger import Logger
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = Logger(__file__)


class TIMEBANDTrainer:
    def __init__(
        self,
        config: dict,
        dataset: TIMEBANDDataset,
        models: TIMEBANDModel,
        metric: TIMEBANDMetric,
        dashboard: TIMEBANDDashboard,
        device: torch.device,
    ) -> None:
        # Set device
        self.device = device

        self.dataset = dataset
        self.models = models
        self.metric = metric
        self.dashboard = dashboard

        # Set Config
        config = self.set_config(config=config)

    def set_config(self, config: dict = None) -> dict:
        """
        Configure settings related to the data set.

        params:
            config: Trainer configuration dict
                `config['trainer']`
        """

        # Train option
        self.lr = config["learning_rate"]["base"]
        self.lr_decay = config["learning_rate"]["decay"]
        self.lr_gammaG = config["learning_rate"]["gammaG"]
        self.lr_gammaD = config["learning_rate"]["gammaD"]

        self.trainer_config = config["epochs"]
        self.epochs = config["epochs"]["iter"]
        self.base_epochs = config["epochs"]["base"]
        self.iter_epochs = config["epochs"]["iter"]
        self.iter_critic = config["epochs"]["critic"]

        # Models options
        self.reload_option = config["models"]["reload"]
        self.reload_counts = 0
        self.reload_interval = config["models"]["reload_interval"]
        self.save_interval = config["models"]["save_interval"]

        # FIXME Test config
        self.amplifier = config["amplifier"]
        self.print_verbose = config["print"]["verbose"]
        self.print_interval = config["print"]["interval"]

    def model_update(self, epoch: int, score: float) -> None:
        # Best Model Save options
        if score < self.models.best_score:
            self.reload_counts = -1
            self.models.best_score = score
            self.models.save(f"{score:.3f}", best=True)

        # Periodic Save options
        if (epoch + 1) % self.save_interval == 0:
            self.models.save()
        
        # Best Model Reload options
        if self.reload_option:
            self.reload_counts += 1
            if self.reload_counts >= self.reload_interval:
                self.reload_counts = 0
                logger.info(f" - Learning rate decay {self.lr} to {self.lr * self.lr_decay}")
                self.lr *= self.lr_decay
                self.models.load("BEST")

    def train(self, trainset: DataLoader, validset: DataLoader) -> None:
        logger.info("Train the model")

        # Score plot
        train_score_plot = []
        valid_score_plot = []
        EPOCHS = self.base_epochs + self.iter_epochs
        for epoch in range(self.base_epochs, EPOCHS):
            # Model Settings
            paramD, lrD = self.models.netD.parameters(), self.lr * self.lr_gammaD
            paramG, lrG = self.models.netG.parameters(), self.lr * self.lr_gammaG
            self.optimD = RMSprop(paramD, lr=lrD)
            self.optimG = RMSprop(paramG, lr=lrG)

            # Prediction
            self.data_idx = 0
            self.pred_initate()

            # Dashboard
            self.dashboard.init_figure()

            # Train Section
            losses = init_loss()
            train_tqdm = tqdm(trainset, loss_info("Train", epoch, losses))
            train_score = self.train_step(train_tqdm, epoch, training=True)
            # train_score_plot.append(train_score)

            # Valid Section
            losses = init_loss()
            valid_tqdm = tqdm(validset, loss_info("Valid", epoch))
            valid_score = self.train_step(valid_tqdm, epoch, training=False)
            # valid_score_plot.append(valid_score)

            self.model_update(epoch, valid_score)

            # Dashboard
            self.dashboard.clear_figure()

        self.models.load("BEST")
        self.models.save(best=True)

    def train_step(self, tqdm, epoch, training=True):
        def discriminate(x):
            return self.models.netD(x).to(self.device)

        def generate(x):
            # FIXME
            # 현재 Obeserved Len은 forecast Len보다 크거나 같아야함.
            fake_y = self.models.netG(x)[:, : self.dataset.forecast_len]
            return fake_y.to(self.device)

        i = 0
        losses = init_loss()
        amplifier = self.amplifier
        TAG = "Train" if training else "Valid"

        for i, data in enumerate(tqdm):
            # #######################
            # Critic
            # #######################
            if training:
                for _ in range(self.iter_critic):
                    # Data Load
                    true_x, true_y = self.dataset.get_random()
                    fake_y = generate(true_x)

                    # Optimizer initialize
                    self.optimD.zero_grad()

                    Dx = discriminate(true_y)
                    Dy = discriminate(fake_y)

                    errD_real = self.metric.GANloss(Dx, target_is_real=True)
                    errD_fake = self.metric.GANloss(Dy, target_is_real=False)
                    errGP = self.metric.grad_penalty(fake_y, true_y)

                    errD = errD_real + errD_fake + errGP
                    errD.backward()
                    self.optimD.step()

            # Data
            true_x = data["encoded"].to(self.device)
            true_y = data["decoded"].to(self.device)

            # #######################
            # Discriminator Training
            # #######################
            # Optimizer initialize
            self.optimD.zero_grad()
            self.optimG.zero_grad()

            fake_y = generate(true_x)
            Dx = discriminate(true_y)
            Dy = discriminate(fake_y)

            errD_real = self.metric.GANloss(Dx, target_is_real=True)
            errD_fake = self.metric.GANloss(Dy, target_is_real=False)

            losses["D"] += errD_real + errD_fake
            losses["Dr"] += errD_real
            losses["Df"] += errD_fake

            if training:
                errD = errD_real + errD_fake
                errD.backward()
                self.optimD.step()

            # #######################
            # Generator Trainining
            # #######################
            fake_y = generate(true_x)
            Dy = self.models.netD(fake_y)
            errG_ = self.metric.GANloss(Dy, target_is_real=False)
            errl1 = self.metric.l1loss(fake_y, true_y)
            errl2 = self.metric.l2loss(fake_y, true_y)
            errGP = self.metric.grad_penalty(fake_y, true_y)
            errG = errG_ + errl1 + errl2 + errGP

            losses["G"] += errG_
            losses["l1"] += errl1
            losses["l2"] += errl2
            losses["GP"] += errGP

            if training:
                errG.backward()
                self.optimG.step()

            # #######################
            # Scoring
            # #######################
            pred_y = self.dataset.denormalize(fake_y.cpu())
            (batchs, forecast_len, target_dims) = true_y.shape
            self.pred_concat(pred_y)

            real_y = self.dataset.forecast[
                self.data_idx : self.data_idx + self.preds.shape[0]
            ]
            self.data_idx += batchs

            losses["Score"] += self.metric.NMAE(self.preds, real_y).detach().numpy()
            losses["RMSE"] += self.metric.RMSE(self.preds, real_y).detach().numpy()
            losses["Score_raw"] += self.metric.NMAE(self.preds, real_y).detach().numpy()
            nme = self.metric.NME(self.preds * self.amplifier, real_y).detach().numpy()
            losses["NME"] += nme

            # if training and i > 30:
            #     amplifier += nme * amplifier * (batchs / self.dataset.data_length) * 0.1

            # Losses Log
            tqdm.set_description(loss_info(TAG, epoch, losses, i))
            if not training:
                self.dashboard.visualize(batchs, real_y, self.preds, self.stds)

        # if training:
        # print(f"Amplifier {self.amplifier:2.5f}, {amplifier:2.5f}")
        # self.amplifier = self.amplifier + (amplifier - self.amplifier) * 0.1

        return losses["Score"] / (i + 1)

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

    def pred_concat(self, pred):
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


def loss_info(process, epoch, losses=None, i=0):
    if losses is None:
        losses = init_loss()

    score = f"{losses['Score'] / (i + 1):7.5f}"
    score = colorstr(score) if process == "Valid" else score
    return (
        f"[{process} e{epoch + 1:4d}]"
        f"Score {losses['Score_raw']/(i+1):7.5f} / {score} / "
        f"{losses['RMSE']/(i+1):7.2f} / "
        f"{losses['NME']/(i+1):7.4f}  ("
        f"D {losses['D']/(i+1):6.3f} "
        f"(R {losses['Dr']/(i+1):6.3f}, "
        f"F {losses['Df']/(i+1):6.3f}) "
        f"G {losses['G']/(i+1):6.3f} "
        f"L1 {losses['l1']/(i+1):6.3f} "
        f"L2 {losses['l2']/(i+1):6.3f} "
        f"GP {losses['GP']/(i+1):6.3f} "
    )


def init_loss() -> dict:
    return {
        "G": 0,
        "D": 0,
        "Dr": 0,
        "Df": 0,
        "l1": 0,
        "l2": 0,
        "GP": 0,
        "NME": 0,
        "RMSE": 0,
        "Score": 0,
        "Score_raw": 0,
    }
