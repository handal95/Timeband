import pickle
from source.args import CLIParser
from source.core import Timeband
from source.utils.initiate import seeding, load_config, setting_path


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
    
    CORE_PATH = "models/sample.pkl"
    if config["train_mode"]:
        model.fit()

    """
    3. 모델 예측
    
    """

    if config["clean_mode"]:    
        line, band = model.predicts()



if __name__ == "__main__":
    main()
