import torch
import torch.nn as nn


class TIMEBANDMetric:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.mse = nn.MSELoss().to(device)
        self.init_score()

    def init_score(self):
        self.scoree = 0
        self.nme = 0
        self.nmae = 0
        self.rmse = 0
        return self.score()

    def NME(self, true: torch.tensor, pred: torch.tensor):
        true, pred = self._ignore_zero(true, pred)

        normalized_error = (true - pred) / true
        normalized_mean_error = torch.mean(normalized_error)
        nme_score = normalized_mean_error.detach().numpy()

        self.nme += nme_score
        return nme_score
    
    def SCORE(self, true: torch.tensor, pred: torch.tensor):
        true, pred = self._ignore_zero(true, pred)

        normalized_error = (true - pred) / true
        normalized_abs_error = torch.abs(normalized_error)
        normalized_mean_abs_error = torch.mean(normalized_abs_error)
        nmae_score = normalized_mean_abs_error.detach().numpy()

        self.scoree += nmae_score
        return nmae_score
    
    def NMAE(self, true: torch.tensor, pred: torch.tensor):
        true, pred = self._ignore_zero(true, pred)

        normalized_error = (true - pred) / true
        normalized_abs_error = torch.abs(normalized_error)
        normalized_mean_abs_error = torch.mean(normalized_abs_error)
        nmae_score = normalized_mean_abs_error.detach().numpy()

        self.nmae += nmae_score
        return nmae_score

    def RMSE(self, true: torch.tensor, pred: torch.tensor):
        true, pred = self._ignore_zero(true, pred)

        mean_squared_error = self.mse(true, pred) + 1e-8
        root_mean_squared_error = torch.sqrt(mean_squared_error)
        rmse_score = root_mean_squared_error.detach().numpy()

        self.rmse += rmse_score
        return rmse_score

    def _ignore_zero(self, true: torch.tensor, pred: torch.tensor):
        target = torch.where(true != 0)
        true = true[target]
        pred = pred[target]
        return true, pred

    def score(self, i: int = 0):
        score = {
            "SCORE": f"{self.scoree / (i + 1): 7.5f}",
            "NME": f"{self.nme  / (i + 1):6.3f}",
            "NMAE": f"{self.nmae / (i + 1):7.5f}",
            "RMSE": f"{self.rmse / (i + 1):6.2f}",
        }
        return score
