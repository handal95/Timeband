import numpy as np

from utils.logger import Logger
from utils.initiate import init_device

import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(
        self,
        in_dim,
        in_seq,
        out_dim,
        out_seq,
        hid_dim=256,
        n_layers=1,
    ):
        super().__init__()
        self.device = init_device()
        self.n_layers = n_layers
        self.hidden_dim = hid_dim
        self.out_dim = out_dim
        self.inseq_len = in_seq
        self.outseq_len = out_seq

        h0_dim = hid_dim // 4
        h1_dim = hid_dim // 2
        h2_dim = hid_dim

        self.lstm0 = nn.LSTM(in_dim, h0_dim, n_layers, batch_first=True).to(self.device)
        self.lstm1 = nn.LSTM(h0_dim, h1_dim, n_layers, batch_first=True).to(self.device)
        self.lstm2 = nn.LSTM(h1_dim, h2_dim, n_layers, batch_first=True).to(self.device)

        self.lstm0.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        self.seq_layer = nn.Sequential(nn.Linear(in_seq, out_seq)).to(self.device)
        self.dim_layer = nn.Sequential(nn.Linear(h2_dim, out_dim), nn.Tanh()).to(
            self.device
        )

    def forward(self, input):
        h0_dim = self.hidden_dim // 4
        batch_size = input.size(0)

        h_0 = torch.zeros(self.n_layers, batch_size, h0_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, batch_size, h0_dim).to(self.device)

        recurrent_features, _ = self.lstm0(input, (h_0, c_0))
        recurrent_features, _ = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        outputs = recurrent_features
        outputs = self.seq_layer(
            outputs.contiguous().view(batch_size, self.hidden_dim, self.inseq_len)
        )
        outputs = self.dim_layer(
            outputs.contiguous().view(batch_size, self.outseq_len, self.hidden_dim)
        )

        return outputs.to(self.device)


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outpus a probability for each element.

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
        device: device for running model (ex. cuda / cpu)

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_seq, in_dim, out_dim, out_seq, hid_dim=256, n_layers=1):
        super().__init__()
        self.device = init_device()
        self.n_layers = n_layers
        self.in_seq = in_seq
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.out_seq = out_seq

        self.lstm = nn.LSTM(in_dim, hid_dim, n_layers, batch_first=True).to(self.device)
        self.lstm.flatten_parameters()

        self.linear = nn.Sequential(nn.Linear(hid_dim, 1), nn.Sigmoid()).to(self.device)

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(
            recurrent_features.contiguous().view(batch_size * seq_len, self.hid_dim)
        )
        outputs = outputs.view(batch_size, seq_len, self.out_dim)

        return outputs.to(self.device)


class TimebandBase:
    def __init__(self):
        self.device = init_device()
        self.logger = Logger("logs", 0)


class TimeModel(TimebandBase):
    def __init__(
        self,
        encode_dim: int,
        decode_dim: int,
        hidden_dim: int,
        observed_len: int,
        forecast_len: int,
    ):
        super(TimeModel, self).__init__()

        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.hidden_dim = hidden_dim

        self.observed_len = observed_len
        self.forecast_len = forecast_len

        # Path Setting
        self.netD = LSTMDiscriminator(
            in_dim=self.decode_dim,
            in_seq=self.forecast_len,
            hid_dim=self.hidden_dim,
            out_dim=1,
            out_seq=self.forecast_len,
        )
        self.netG = LSTMGenerator(
            in_dim=self.encode_dim,
            in_seq=self.observed_len,
            hid_dim=self.hidden_dim,
            out_dim=self.decode_dim,
            out_seq=self.forecast_len,
        )
        self.logger.info(self.netD)
        self.logger.info(self.netG)

    def pred_initiate(self):
        forecast_len = self.forecast_len
        decode_dim = self.decode_dim

        self.preds = np.empty((forecast_len - 1, forecast_len, decode_dim))
        self.preds[:] = np.nan

    def generate(self, x):
        return self.netG(x).to(self.device)

    def discriminate(self, x):
        return self.netD(x).to(self.device)

    def predicts(self, pred):
        (batch_size, forecast_len, target_dims) = pred.shape
        pred = pred.cpu().detach().numpy()

        nan_shape = np.empty((batch_size, forecast_len, target_dims))
        nan_shape[:] = np.nan

        self.preds = np.concatenate([self.preds[1 - forecast_len :], nan_shape])
        for f in range(forecast_len):
            self.preds[f : batch_size + f, f] = pred[:, f]

        preds = np.nanmedian(self.preds, axis=1)
        std = np.nanstd(self.preds, axis=1)

        for f in range(forecast_len - 1, 0, -1):
            gamma = (forecast_len - f) / (forecast_len - 1)
            std[-f] += std[-f - 1] * gamma

        lower = preds - 2 * std
        upper = preds + 2 * std

        return preds, lower, upper

    def adjust(self, output, preds, masks, lower, upper):
        len = preds.shape[0]
        a = self.missing_gamma
        b = self.anomaly_gamma

        for p in range(len):
            value = output[p + 1]

            mmask = masks[p]
            lmask = (value < lower[p]) * (1 - mmask)
            umask = (value > upper[p]) * (1 - mmask)

            value = (1 - mmask) * value + mmask * (a * preds[p] + (1 - a) * output[p])
            value = (1 - lmask) * value + lmask * (b * preds[p] + (1 - b) * value)
            value = (1 - umask) * value + umask * (b * preds[p] + (1 - b) * value)

            output[p + 1] = value

        target = output[1:]
        return target
