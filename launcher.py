import os
import pickle
import pandas as pd

from .source.args import CLIParser
from .source.core import Timeband
from .source.utils.initiate import seeding, load_config, setting_path


seeding(seed=42)


def main():
    """
    1. 모델 생성로직

    """
    # Setting Configuration
    config = load_config(config_path="model.cfg")
    config = CLIParser(config).config
    config = setting_path(config)

    # Model initiating
    model = Timeband(config)

    """
    2. 모델 학습로직

    """
    # DATA
    model.fit()

    CORE_PATH = "models/sample.pkl"
    with open(CORE_PATH, mode="wb") as f:
        pickle.dump(model, f)

    del model
    """
    3. 모델 예측

    """

    with open(CORE_PATH, mode="rb") as f:
        model = pickle.load(f)

    line, band = model.predicts()

    print(line)
    print("=====")
    print(band)

    CORE_PATH = "models/sample.pkl"
    with open(CORE_PATH, mode="wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
