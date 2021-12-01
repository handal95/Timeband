import pickle
from re import sub
import pandas as pd
from source.args import CLIParser
from source.core import Timeband
from source.utils.initiate import seeding, load_config, setting_path
from source.data import TIMEBANDData
from tqdm import tqdm
from torch.utils.data import DataLoader

seeding(seed=42)


def main():

    """
    0. Data 전처리

    """
    OBSERVED_LENGTH = 5
    FORECAST_LENGTH = 5
    Data = TIMEBANDData(
        basedir="data/",
        filename="067160",
        targets=["Close"],
        drops=["Change"],
        fill_timegap=False,
        time_index=["Date"],
        time_encode=["year", "month", "weekday", "day"],
        split_size=0.8,
        observed_len=OBSERVED_LENGTH,
        forecast_len=FORECAST_LENGTH,
    )

    data = Data.init_dataset(index_s=0, index_e=None, force=True)
    trainset, validset = Data.prepare_dataset(data, split=True)

    input("Step 1 >> Input Enter")
    """
    1. 모델 생성로직

    """
    # Setting Configuration
    config = load_config(config_path="model.cfg")
    config = CLIParser(config).config
    config = setting_path(config)

    # Model initiating
    model = Timeband(config)
    dataloader = DataLoader(trainset, batch_size=1)
    for epoch in range(10):
        for i, data in tqdm(dataloader, desc=f"Epoch {epoch:3d}"):
            x = data

    input("Step 2 >> Input Enter")
    """
    2. 모델 학습로직
    
    """
    # DATA

    CORE_PATH = "models/sample.pkl"
    if config["train_mode"]:
        model.fit()

    input("Step 3 >> Input Enter")
    """
    3. 모델 예측
    
    """

    if config["clean_mode"]:
        line, band = model.predicts()

    """
    4. 배치 예측
    
    """
    Data = TIMEBANDData(
        basedir="data/",
        filename="067160",
        targets=["Close"],
        drops=["Change"],
        fill_timegap=False,
        time_index=["Date"],
        time_encode=["year", "month", "weekday", "day"],
        split_size=0.8,
        observed_len=OBSERVED_LENGTH,
        forecast_len=FORECAST_LENGTH,
    )
    subdata = pd.read_csv("data/origin/067160.csv", parse_dates=["Date"])
    subdata.set_index(["Date"], inplace=True)
    subdata = subdata.iloc[-15:]

    dataset = Data.prepare_dataset(subdata, split=False)
    print(dataset.shape())


if __name__ == "__main__":
    main()
