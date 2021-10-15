# import os
# import yaml
import argparse
from argparse import Namespace


class Parser:
    def __init__(self, config: dict) -> None:
        opt = self.get_parser(config)

        self.config = self.set_config(config, opt)

    def get_parser(self, config) -> Namespace:
        parser = argparse.ArgumentParser(description="** BandGan CLI **")
        parser.set_defaults(function=None)

        # DASHBOARD
        parser.add_argument(
            "-v",
            "--visualize",
            type=bool,
            help="Visualize options",
            default=config["dashboard"]["visualize"],
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

    def set_config(self, config: dict, parser: Namespace) -> dict:
        # -v, --visual
        config["dashboard"]["width"] = parser.dashboard_width
        config["dashboard"]["height"] = parser.dashboard_height
        config["dashboard"]["visualize"] = parser.visualize

        return config

    def __str__(self):
        return str(self.opt)
