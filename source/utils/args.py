import argparse
from argparse import Namespace


class ArgParser:
    def __init__(self, config: dict) -> None:
        opt = self.get_parser(config)

        self.config = self.set_config(config, opt)

    def set_config(self, config: dict, parser: Namespace) -> dict:
        config["train_mode"] = parser.train_mode
        config["clean_mode"] = parser.clean_mode
        config["preds_mode"] = parser.preds_mode

        config["core"]["batch_size"] = max(parser.batch_size, 1)

        config["trainer"]["epochs"] = parser.epochs

        config["dashboard"]["width"] = parser.dashboard_width
        config["dashboard"]["height"] = parser.dashboard_height
        config["dashboard"]["vis_opt"] = parser.vis_opt

        return config

    def get_parser(self, config) -> Namespace:
        parser = argparse.ArgumentParser(description="** TIMEBAND CLI **")
        parser.set_defaults(function=None)

        # Launcher - mode
        parser.add_argument(
            "-tm",
            "--train_mode",
            type=self.str2bool,
            help="If True, Do the train",
            default=config["train_mode"],
        )
        parser.add_argument(
            "-cm",
            "--clean_mode",
            type=self.str2bool,
            help="If True, Do the clean",
            default=config["clean_mode"],
        )
        parser.add_argument(
            "-pm",
            "--preds_mode",
            type=self.str2bool,
            help="If True, Do the run",
            default=config["preds_mode"],
        )

        # CORE OPTION
        parser.add_argument(
            "-b",
            "--batch_size",
            type=int,
            help="train epochs",
            default=config["core"]["batch_size"],
        )

        # TRAIN OPTION
        parser.add_argument(
            "-ep",
            "--epochs",
            type=int,
            help="train epochs",
            default=config["trainer"]["epochs"],
        )

        # DASHBOARD
        parser.add_argument(
            "-v",
            "--vis_opt",
            type=self.str2bool,
            help="Visualize options",
            default=config["dashboard"]["vis_opt"],
        )
        parser.add_argument(
            "-dw",
            "--dashboard_width",
            type=int,
            help="size of dashboard width",
            default=config["dashboard"]["width"],
        )
        parser.add_argument(
            "-dh",
            "--dashboard_height",
            type=int,
            help="size of dashboard height",
            default=config["dashboard"]["height"],
        )
        return parser.parse_args()

    def str2bool(self, value: str):
        if isinstance(value, bool):
            return value

        if value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False

        raise argparse.ArgumentTypeError("Boolean value expected.")

    def __str__(self):
        return str(self.opt)