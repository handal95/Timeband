import torch
import numpy as np

from tqdm import tqdm
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.color import colorstr
from TIMEBAND.loss import TIMEBANDLoss
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
        self.amplifier_scope = config["amplifier_scope"]
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
                logger.info(
                    f" - Learning rate decay {self.lr} to {self.lr * self.lr_decay}"
                )
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
            self.optimD = Adam(paramD, lr=lrD)
            self.optimG = Adam(paramG, lr=lrG)

            # Prediction
            self.data_idx = 0
            self.pred_initate()

            # Dashboard
            self.dashboard.init_figure()

            # Train Section
            train_score = self.train_step(epoch, trainset, training=True)
            train_score_plot.append(train_score)

            # Valid Section
            valid_score = self.train_step(epoch, validset, training=False)
            valid_score_plot.append(valid_score)

            self.model_update(epoch, valid_score)

            # Dashboard
            self.dashboard.clear_figure()

        self.models.load("BEST")
        self.models.save(best=True)

    def train_step(self, epoch, dataset, training=True):
        def discriminate(x):
            return self.models.netD(x).to(self.device)

        def generate(x):
            return self.models.netG(x)[:, : self.forecast_len].to(self.device)

        amplifier = self.amplifier

        losses = self.losses.init_loss()
        score = self.metric.init_score()
        tqdm_ = tqdm(dataset, desc(training, epoch, score, losses))
        for i, data in enumerate(tqdm_):
            # #######################
            # Critic & Optimizer init
            # #######################
            if training:
                paramD, lrD = self.models.netD.parameters(), self.lr * self.lr_gammaD
                optimD = RMSprop(paramD, lr=lrD)
                for _ in range(self.iter_critic):
                    # Load Random Sample Data
                    true_x, true_y = self.dataset.get_random()
                    fake_y = generate(true_x)

                    # Optimizer initialize
                    optimD.zero_grad()

                    Dy = discriminate(true_y)
                    DGx = discriminate(fake_y)

                    errD = self.losses.dis_loss(true_y, fake_y, Dy, DGx, critic=True)
                    errD.backward()
                    optimD.step()

            self.optimD.zero_grad()
            self.optimG.zero_grad()

            # #######################
            # Load Data
            # #######################
            true_x = data["encoded"].to(self.device)
            true_y = data["decoded"].to(self.device)
            (batchs, forecast_len, target_dims) = true_y.shape

            # #######################
            # Discriminator Training
            # #######################
            fake_y = generate(true_x)
            Dy = discriminate(true_y)
            DGx = discriminate(fake_y)
            if training:
                errD = self.losses.dis_loss(true_y, fake_y, Dy, DGx)
                errD.backward()
                self.optimD.step()
            else:
                with torch.no_grad():
                    self.losses.dis_loss(true_y, fake_y, Dy, DGx)

            # #######################
            # Generator Trainining
            # #######################
            fake_y = generate(true_x)
            DGx = self.models.netD(fake_y)
            if training:
                errG = self.losses.gen_loss(true_y, fake_y, DGx)
                errG.backward()
                self.optimG.step()
            else:
                with torch.no_grad():
                    self.losses.gen_loss(true_y, fake_y, DGx)

            # #######################
            # Scoring
            # #######################
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
            nme = self.metric.NME(reals, preds * self.amplifier)
            if training and i > self.amplifier_scope:
                amplifier += nme * amplifier * (batchs / self.dataset.data_length) * 0.1
            
            losses = self.losses.loss(i)
            score = self.metric.score(i)

            # Losses Log
            tqdm_.set_description(desc(training, epoch, score, losses))
            # if not training:
            self.dashboard.visualize(batchs, reals, self.preds * self.amplifier, self.stds)

        if training:
            self.amplifier = self.amplifier + (amplifier - self.amplifier) * 0.8
        elif (epoch + 1) % self.print_interval == 0:
            print(f"[ Amplifier ] {self.amplifier:2.5f}")

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
