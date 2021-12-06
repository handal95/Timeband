import numpy as np
import pandas as pd

from .utils.logger import Logger
from .utils.initiate import init_device

from tqdm import tqdm
from source.data import TIMEBANDData
from source.models import TimebandModel
from source.metrics import TIMEBANDMetric
from source.losses import TimebandLoss
from torch.optim import RMSprop, Adam


class Timeband:
    VERSION = "Timeband v2.3.0"

    def __init__(
        self,
        datadir: str,
        filename: str,
        observed_len: int,
        forecast_len: int,
        l1_weights: int,
        l2_weights: int,
        gp_weights: int,
    ):
        super(Timeband).__init__()
        self.device = init_device()
        self.logger = Logger("logs", 0)

        self.epochs = 0
        self.best_score = 10.0

        self.datadir = datadir
        self.filename = filename

        self.observed_len = observed_len
        self.forecast_len = forecast_len

        self.Metric = TIMEBANDMetric()
        self.Losses = TimebandLoss(
            l1_weights=l1_weights, l2_weights=l2_weights, gp_weights=gp_weights
        )
        import os
        
        d = pd.read_csv(os.path.join(self.datadir,"origin", f"{self.filename}.csv"))
        d.drop(columns=['Date', 'KOSPI', 'KOSDAQ'], inplace=True)
        self.Data = TIMEBANDData(
            basedir=self.datadir,
            filename=self.filename,
            targets=d.columns,
            drops=[],
            fill_timegap=False,
            time_index=["Date"],
            time_encode=["year", "month", "weekday", "day"],
            split_size=1.0,
            observed_len=self.observed_len,
            forecast_len=self.forecast_len,
        )
        print(self.Data.encode_dims)
        print(self.Data.decode_dims)
        self.Model = TimebandModel(
            encode_dim=self.Data.encode_dims,
            decode_dim=self.Data.decode_dims,
            hidden_dim=256,
            observed_len=self.observed_len,
            forecast_len=self.forecast_len,
        )

    def init_optimizer(self, lr_D: float, lr_G: float):
        # Optimizer Setting
        netD, netG = self.Model.netD, self.Model.netG

        self.optimD = Adam(netD.parameters(), lr=lr_D)
        self.optimG = Adam(netG.parameters(), lr=lr_G)
        self.optimC = RMSprop(netD.parameters(), lr=lr_D)

    def critic(self, trainset, critics):
        self.Losses.init_loss()
        for critic in range(critics):
            self.optimC.zero_grad()

            true_x, true_y = self.Data.get_random(trainset)
            true_x = true_x.to(self.device)
            true_y = true_y.to(self.device)

            fake_y = self.Model.generate(true_x)

            Dy = self.Model.discriminate(true_y)
            DGx = self.Model.discriminate(fake_y)

            errC = self.Losses.dis_loss(Dy, DGx)
            errC.backward()

            self.optimC.step()

    def train_step(self, dataloader, training=False):
        self.Model.pred_initiate()
        self.Metric.init_score()
        self.Losses.init_loss()

        _tqdm = tqdm(dataloader)
        for i, data in enumerate(_tqdm):
            # ##########
            # Discriminator Training
            # ##########
            if training:
                self.optimD.zero_grad()

            true_x, true_y = data
            true_x = true_x.to(self.device)
            true_y = true_y.to(self.device)
            
            fake_y = self.Model.generate(true_x)

            Dy = self.Model.discriminate(true_y)
            DGx = self.Model.discriminate(fake_y)

            errD = self.Losses.dis_loss(Dy, DGx)
            if training:
                errD.backward()
                self.optimD.step()

            # ##########
            # Generator Training
            # ##########
            if training:
                self.optimG.zero_grad()

            fake_y = self.Model.generate(true_x)
            DGx = self.Model.discriminate(fake_y)

            errG = self.Losses.gen_loss(true_y, fake_y, DGx)
            if training:
                errG.backward()
                self.optimG.step()

            # ##########
            # Band Process
            # ##########
            pred_y = self.Data.denormalize(fake_y)
            preds, lower, upper = self.Model.predicts(pred_y)

            batchs, pred_len, target_dim = pred_y.shape
            reals = self.Data.forecast[self.idx : self.idx + pred_len]
            masks = self.Data.missing_decode[self.idx : self.idx + pred_len]
            self.Metric.scoring(reals, preds, masks)

            scores = self.Metric.score(i)
            process = "Train" if training else "Valid"
            _tqdm.set_description(
                f"{process} [Epoch {self.epochs:3d}] NMAE: {scores['NMAE']} RMSE: {scores['RMSE']} NME: {scores['NME']}"
            )
            self.idx = self.idx + batchs

        return float(scores["NMAE"])

    def predict(self, dataloader):
        self.Model.pred_initiate()

        _tqdm = tqdm(dataloader)
        outputs = np.zeros((self.forecast_len, self.Data.decode_dims))
        lower_bands = outputs.copy()
        upper_bands = outputs.copy()
        for i, data in enumerate(_tqdm):
            true_x, true_y = data
            true_x = true_x.to(self.device)

            fake_y = self.Model.generate(true_x)
            pred_y = self.Data.denormalize(fake_y)
            preds, lower, upper = self.Model.predicts(pred_y)

            _tqdm.set_description(f"Preds [Epoch {self.epochs:3d}]")

            outputs = np.concatenate([outputs[: 1 - self.forecast_len], preds])
            lower_bands = np.concatenate([lower_bands[: 1 - self.forecast_len], lower])
            upper_bands = np.concatenate([upper_bands[: 1 - self.forecast_len], upper])

        lower_cols = [f"{x}_lower" for x in self.Data.targets]
        upper_cols = [f"{x}_upper" for x in self.Data.targets]
        outputs_df = pd.DataFrame(outputs, columns=self.Data.targets)

        bands_df = pd.concat(
            [
                pd.DataFrame(lower_bands, columns=lower_cols),
                pd.DataFrame(upper_bands, columns=upper_cols),
            ],
            axis=1,
        )

        return outputs_df, bands_df

    def is_best(self, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            return True
        return False
