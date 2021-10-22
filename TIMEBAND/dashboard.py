import numpy as np
import matplotlib.pyplot as plt

from utils.color import COLORS
from TIMEBAND.dataset import TIMEBANDDataset

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)


class TIMEBANDDashboard:
    def __init__(self, config: dict, dataset: TIMEBANDDataset) -> None:
        # Set Config
        self.set_config(config=config)

        # Dataset
        self.time_idx = 0
        self.dataset = dataset

        self.times = dataset.times
        self.observed = dataset.observed
        self.target_cols = dataset.targets
        self.target_dims = len(self.target_cols)

        self.observed_len = dataset.observed_len
        self.forecast_len = dataset.forecast_len

        # Figure and Axis
        self.fig, self.axes = None, None

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
        self.band_width = config["band_width"]
        self.feats_by_rows = config["features_by_rows"]
        self.xinterval = config["xinterval"]

        self.height = config["height"]
        self.width = config["width"]

    def init_figure(self) -> tuple:
        if self.visual is False:
            return

        self.time_idx = 0

        # Config subplots
        nrows = 2 # 2 + (self.target_dims - 1) // self.feats_by_rows
        ncols = 1
        size = (self.width, self.height)

        fig, axes = plt.subplots(nrows, ncols, figsize=size, clear=True, sharex=True)
        # fig.tight_layout()
        # axes[0].set_title("TARGET FEATURES")

        for i, ax in enumerate(axes):
            idx_s = i * self.feats_by_rows
            idx_e = idx_s + min(self.target_dims - idx_s, self.feats_by_rows)
            subplot_title = f"TIMEBAND-Band Feature {idx_s} to {idx_e}"
            ax.set_title(subplot_title)

        self.reals = self.observed
        self.preds = self.observed
        self.lower = self.observed
        self.upper = self.observed

        self.fig, self.axes = fig, axes

    def visualize(self, batchs, real_data, preds, std):
        if self.visual is False:
            return

        self.reals = np.concatenate([self.reals[: 1 - self.forecast_len], real_data])
        self.preds = np.concatenate([self.preds[: 1 - self.forecast_len], preds])
        self.lower = np.concatenate(
            [self.lower[: 1 - self.forecast_len], preds - self.band_width * std]
        )
        self.upper = np.concatenate(
            [self.upper[: 1 - self.forecast_len], preds + self.band_width * std]
        )

        for batch in range(batchs):
            fig, axes = self.reset_figure()
            # 하단 그래프
            START = max(self.time_idx - self.scope, 0)
            PIVOT = START + min(self.scope, self.time_idx)
            OBSRV = PIVOT + self.observed_len
            FRCST = OBSRV + self.forecast_len

            xticks = np.arange(START, FRCST + 1)
            true_ticks = np.arange(START, OBSRV)
            pred_ticks = np.arange(START, FRCST)
            timelabel = [self.times[x] for x in xticks]
            col = 0
            for i, ax in enumerate(axes):
                ax.set_xticks(xticks[:: self.xinterval])
                ax.set_xticklabels(timelabel[:: self.xinterval], rotation=30)

                idx_s = i * self.feats_by_rows
                idx_e = idx_s + min(self.target_dims - idx_s, self.feats_by_rows)

                ax.axvline(START, color="black")
                ax.axvline(PIVOT, color="blue")
                ax.axvline(OBSRV - 1, color="red")
                ax.axvline(FRCST - 1, color="black")

                ax.axvspan(PIVOT, OBSRV - 1, alpha=0.1, label="Observed")
                ax.axvspan(OBSRV - 1, FRCST - 1, alpha=0.1, color="r", label="Forecast")

                for col in range(idx_s, idx_e):
                    feature_label = self.target_cols[col]
                    color = COLORS[col]
                    ax.plot(
                        true_ticks,
                        self.reals[START:OBSRV, col],
                        color=color,
                    )
                    ax.plot(
                        pred_ticks,
                        self.preds[START:FRCST, col],
                        alpha=0.2,
                        linewidth=5,
                        color=color,
                        label=f"Pred {feature_label}",
                    )
                    ax.fill_between(
                        pred_ticks,
                        self.lower[START:FRCST, col],
                        self.upper[START:FRCST, col],
                        alpha=0.2,
                        color=color,
                        label="Normal Band",
                    )
                    col += 1
                ax.legend(loc="lower left")
                ax.relim()
            self.time_idx += 1
            self.show_figure()



    def reset_figure(self):
        # Clear previous figure
        for i in range(len(self.axes)):
            self.axes[i].clear()
            self.axes[i].set_ylim(auto=True)

        return self.fig, self.axes

    def show_figure(self) -> None:
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear_figure(self) -> None:
        if self.visual is False:
            return

        plt.close("all")
        plt.clf()

        del self.preds
        del self.lower
        del self.upper
        self.fig = None
        self.axes = None
