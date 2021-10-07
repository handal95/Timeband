import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from TIMEBAND.dataset import TIMEBANDDataset

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rc('font', family='Malgun Gothic')


class TIMEBANDDashboard:
    def __init__(self, config: dict, dataset: TIMEBANDDataset) -> None:
        # Set Config
        self.set_config(config=config)

        # Dataset
        self.dataset = dataset
        self.index = 0
        self.origin_data = dataset.origin_data
        self.origin_cols = dataset.origin_cols
        self.origin_dims = len(self.origin_cols)

        self.target_data = dataset.target_data
        self.target_cols = dataset.target_cols
        self.target_dims = len(self.target_cols)

        self.timestamp = dataset.timestamp

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
        
        self.height = config["height"]
        self.width = config["width"]
        

    def init_figure(self) -> tuple:
        # Config subplots
        nrows = 2 + (self.origin_dims - 1) // self.feats_by_rows
        ncols = 1
        size = (self.width, self.height)

        plt.title("주가 데이터 Dashboard")
        fig, axes = plt.subplots(nrows, ncols, figsize=size)
        # fig.tight_layout()

        axes[0].set_title("TARGET FEATURES")

        for i, ax in enumerate(axes[1:]):
            idx_s = i * self.feats_by_rows
            idx_e = idx_s + min(self.origin_dims - idx_s, self.feats_by_rows)
            subplot_title = f"TIMEBAND-Band Feature {idx_s} to {idx_e}"
            ax.set_title(subplot_title)

        self.fig, self.axes = fig, axes

    def visualize(self, batchs, reals, preds):
        if self.visual is False:
            return

        reals = reals.detach().numpy()
        preds = preds.detach().numpy()
        for batch in range(batchs):
            fig, axes = self.reset_figure()

            self.index += 1
            BASE = max(0, self.index - self.observed_len)
            UPTO = self.index + self.observed_len
            PRED_BASE = UPTO - BASE #  - self.forecast_len

            # target_data = self.target_data[]
            axes[0].plot(self.target_data[BASE:UPTO], label=f"REAL")
            
            for forecast in range(self.forecast_len):
                last_observed = self.target_data[UPTO - self.forecast_len + forecast: UPTO + forecast - self.forecast_len + 1]
                preds_plot = np.concatenate([last_observed, preds[forecast + batchs]])
                xrange = (
                    PRED_BASE + np.arange(forecast, self.forecast_len + forecast + 1) - self.forecast_len
                )
                axes[0].plot(xrange, preds_plot, alpha=0.2, label=f"{forecast}")

            # 하단 그래프
            base = max(0, self.index - self.observed_len)
            upto = self.index + self.observed_len + self.forecast_len
            for i, ax in enumerate(axes[1:]):
                idx_s = i * self.feats_by_rows
                idx_e = idx_s + min(self.origin_dims - idx_s, self.feats_by_rows)
                # Observed window
                OBSERVED = max(0, self.index - self.observed_len)
                OBSERVED_UPTO = UPTO - 1
                FORECAST_UPTO = UPTO + self.forecast_len - 1

                #+ self.observed_len #  self.index + self.observed_len
                ax.axvspan(OBSERVED, OBSERVED_UPTO, alpha=0.2, label="Scope")

                # Forecast window
                ax.axvspan(OBSERVED_UPTO, FORECAST_UPTO, color='r', alpha=0.1, label="Forecast")
                # axes[i].axvspan(FORECAST_BASE, base + self.scope + self.forecast_len, alpha=0.1)

                # Origin data
                for idx in range(idx_s, idx_e):
                    feature_label = self.origin_cols[idx]
                    alpha = 1.0 if feature_label in self.target_cols else 0.2
                    ax.plot(self.origin_data[:UPTO, idx], label=feature_label, alpha=alpha)

                ax.legend(loc="center left")
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
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
