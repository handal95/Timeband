import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from TIMEBAND.dataset import TIMEBANDDataset

plt.rcParams["font.family"] = "Malgun Gothic"


class TIMEBANDDashboard:
    def __init__(self, config: dict, dataset: TIMEBANDDataset) -> None:
        # Set Config
        self.set_config(config=config)

        # Dataset
        self.dataset = dataset
        self.index = 0
        self.origin_data = dataset.origin_data
        self.origin_cols = dataset.origin_cols
        self.target_data = dataset.target_data
        self.target_cols = dataset.target_cols

        self.timestamp = dataset.timestamp
        self.decode_dims = dataset.decode_dims

        self.observed_len = dataset.observed_len
        self.forecast_len = dataset.forecast_len

        # Figure and Axis
        self.fig, self.axes = None, None

        # Data
        self.true_data = None

    def set_config(self, config: dict) -> None:
        """
        Configure settings related to the data set.

        params:
            config: Dashboard configuration dict
                `config['dashboard_cfg']`
        """

        # Data file configuration
        self.visual = config["visualize"]
        self.scope = config["scope"]
        self.feats_by_rows = config["features_by_rows"]
        self.xinterval = config["xinterval"]

    def init_figure(self) -> tuple:
        # Config subplots
        nrows = 2 + (self.decode_dims - 1) // self.feats_by_rows
        ncols = 1
        size = (25, 10)

        plt.title("주가 데이터 Dashboard")
        fig, axes = plt.subplots(nrows, ncols, figsize=size)
        fig.tight_layout()

        axes[0].set_title("TARGET FEATURES")

        for row in range(1, len(axes)):
            idx_s = (row - 1) * self.feats_by_rows
            idx_e = idx_s + min(self.decode_dims - idx_s, self.feats_by_rows)
            subplot_title = f"TIMEBAND-Band Feature {idx_s} to {idx_e}"
            axes[row].set_title(subplot_title)

        self.fig, self.axes = fig, axes

    def visualize(self, batchs, reals, preds):
        if self.visual is False:
            return

        reals = reals.detach().numpy()
        preds = preds.detach().numpy()
        preds_plots = []
        for batch in range(batchs):
            fig, axes = self.reset_figure()

            self.index += 1
            BASE = self.index
            UPTO = BASE + self.observed_len

            for forecast in range(self.forecast_len):
                preds_plot = np.concatenate(
                    [
                        self.target_data[BASE + self.observed_len + forecast - 1: BASE + self.observed_len + forecast],
                        preds[self.index + batchs - 1, :],
                    ]
                )
                xrange = (
                    np.arange(forecast - 1, self.forecast_len + forecast)
                    + self.observed_len
                )
                axes[0].plot(xrange, preds_plot, label=f"{forecast}")

            axes[0].plot(
                self.target_data[BASE : UPTO + self.forecast_len - 1], label=f"REAL"
            )

            # 하단 그래프
            base = max(0, self.observed_len - self.scope)
            upto = self.index + self.observed_len + self.forecast_len
            for i in range(1, len(axes)):
                idx_s = (i - 1) * self.feats_by_rows
                idx_e = idx_s + min(self.decode_dims - idx_s, self.feats_by_rows)

                axes[i].axvspan(base, base + self.scope, alpha=0.2)
                axes[i].axvspan(
                    base + self.scope, base + self.scope + self.forecast_len, alpha=0.1
                )

                for idx in range(idx_s, idx_e):
                    axes[i].plot(self.origin_data[batchs:upto, idx], label=f"{self.target_cols[idx]}")
                axes[i].legend(loc="center left")
            self.show_figure()

    def reset_figure(self):
        # Clear previous figure
        for i in range(len(self.axes)):
            self.axes[i].clear()

        return self.fig, self.axes

    def show_figure(self):
        base = max(0, self.observed_len - self.scope)

        xticks = np.arange(base + self.scope)
        xlabels = self.dataset.timestamp[: base + self.scope]

        plt.xticks(xticks[:: self.xinterval], xlabels[:: self.xinterval], rotation=30)
        # plt.legend()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
