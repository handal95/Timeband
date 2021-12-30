import os
import pickle
import pandas as pd
from torch.utils.data import DataLoader

from source.core import Timeband
from source.utils.initiate import seeding
from typing import List

# seeding(seed=42)


def get_path(dirname: str, filename: str, postfix: str = "") -> os.path:
    filename = filename if postfix == "" else f"{filename}_{postfix}"
    filepath = os.path.join(dirname, f"{filename}.pkl")
    return filepath


def load_core(core_path):
    with open(core_path, "rb") as f:
        core = pickle.load(f)

    return core


def save_core(core, core_path, best: bool = False):
    if best:
        print(f"Best Model is Saved at {core_path}")

    with open(core_path, mode="wb") as f:
        pickle.dump(core, f)


def main(FILE_NAME: str, TARGETS: List[str]):
    """
    0. Core 불러오기

    """
    MODEL_PATH = "models/"
    OBSERVED_LEN = 5
    FORECAST_LEN = 3
    os.mkdir(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None

    try:
        CORE_PATH = get_path(MODEL_PATH, FILE_NAME, postfix="best")
        Core = load_core(CORE_PATH)
    except FileNotFoundError:
        Core = Timeband(
            datadir="data/",
            filename=FILE_NAME,
            targets=TARGETS,
            observed_len=OBSERVED_LEN,
            forecast_len=FORECAST_LEN,
            l1_weights=1,
            l2_weights=1,
            gp_weights=1,
        )

    """
    1. 모델 학습

    """
    STEPS = 1
    EPOCHS = 20
    CRITICS = 5
    train_score_plot = []
    valid_score_plot = []
    Core.Data.split_size = 1.0
    dataset = Core.Data.init_dataset(index_s=0, index_e=None)
    Core.init_optimizer(lr_D=1, lr_G=1)

    for step in range(STEPS):
        index_e = None if step + 1 == STEPS else -step
        trainset, validset = Core.Data.prepare_trainset(dataset[:index_e])

        trainloader = DataLoader(trainset, batch_size=256)
        validloader = DataLoader(validset, batch_size=1)

        for epoch in range(EPOCHS):
            Core.idx = Core.observed_len
            Core.critic(trainset, CRITICS)

            # Train Step
            train_score = Core.train_step(trainloader, training=True)
            train_score_plot.append(train_score)

            # Valid Step
            valid_score = Core.train_step(validloader)
            valid_score_plot.append(valid_score)

            Core.epochs += 1
            update = True  # train_score - valid_score < train_score * 0.5
            if update and Core.is_best(valid_score):
                save_core(Core, get_path(MODEL_PATH, FILE_NAME, postfix="best"), best=True)

        if Core.is_best(valid_score):
            save_core(Core, get_path(MODEL_PATH, FILE_NAME, postfix="best"), best=True)


def predict(FILE_NAME: str, TARGETS: List[str], data: pd.DataFrame):
    """
    모델 예측

    """
    MODEL_PATH = "models/"
    os.mkdir(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None

    CORE_PATH = get_path(MODEL_PATH, FILE_NAME, postfix="best")
    Core = load_core(CORE_PATH)

    dataset = Core.Data.prepare_predset(data)

    dataloader = DataLoader(dataset)

    # # Preds Step
    outputs, bands = Core.predict(dataloader)

    return outputs, bands


if __name__ == "__main__":
    main("sample_input", ["aaaaaa_close", "bbbbbb_close", "cccccc_close", "dddddd_close", "eeeeee_close"])
