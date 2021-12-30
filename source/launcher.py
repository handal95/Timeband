"""
TIMEBAND Launcher.

"""
import os
from core import Timeband
from utils.initiate import seeding

from utils.parser import ArgParser
from utils.logger import Logger
from utils.initiate import check_dirs_exist
from utils.files import get_path, load_core
from typing import List
from torch.utils.data import DataLoader
from utils.files import save_core, get_path

seeding()

def launcher(data_file=None, time_index=None, targets=None):
    args = ArgParser(data_file).args
    logger = Logger(args.logfile, args.verbosity)
    logger.debug("Timeband Launcher")
    
    # CLI OPTION
    args.targets = ['value'] if targets is None else targets
    args.data_file = 'A-1' if data_file is None else data_file
    args.time_index = time_index

    # Path check
    rootdir = os.path.dirname(os.getcwd())
    basedirs = [args.data_dir, args.model_dir, args.logs_dir]
    datadirs = ["target/", f"target/{args.data_file}"]

    check_dirs_exist(basepath=rootdir, dirlist=basedirs)
    check_dirs_exist(basepath=args.data_dir, dirlist=datadirs)
    
    core_path = get_path(args.model_dir, args.model_file, postfix="best")

    logger.debug("Timeband Core setting")
    if os.path.exists(core_path):
        Core = load_core(core_path)
    else:
        Core = Timeband(
            datadir=args.data_dir,
            filename=args.data_file,
            targets=args.targets,

            observed_len=args.observed_length,
            forecast_len=args.forecast_length,
            time_index=args.time_index,

            l1_weights=args.l1_weight,
            l2_weights=args.l2_weight,
            gp_weights=args.gp_weight,
        )
        
    # Train option
    STEPS = 1
    EPOCHS = 1
    CRITICS = 5
    train_score_plot = []
    valid_score_plot = []
    Core.Data.split_size = 0.9
    dataset = Core.Data.init_dataset(index_s=0, index_e=None)
    Core.init_optimizer(lr_D=5e-4, lr_G=5e-4)

    for step in range(STEPS):
        index_e = None if step + 1 == STEPS else -step
        trainset, validset = Core.Data.prepare_trainset(dataset[:index_e])

        trainloader = DataLoader(trainset, batch_size=256)
        validloader = DataLoader(validset, batch_size=256)

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
            update = train_score - valid_score < train_score * 0.5
            if update and Core.is_best(valid_score):
                save_core(Core, core_path, best=True)

        if Core.is_best(valid_score):
            save_core(Core, core_path, best=True)

    """
    모델 예측

    """
    logger.info("Prediction")
    core_path = get_path(args.model_dir, args.model_file, postfix="best")
    Core = load_core(core_path)

    dataset = Core.Data.init_dataset(index_s=0, index_e=None)
    dataset = Core.Data.prepare_predset(dataset)
    dataloader = DataLoader(dataset, batch_size=1)

    # # Preds Step
    outputs, bands = Core.predict(dataloader)
    
    print(type(outputs), outputs.shape)
    print(type(bands), bands.shape)
    
    outputs.to_csv("output.csv")
    bands.to_csv("bands.csv")

    return outputs, bands
        

def train(core: Timeband, target):
    pass

def eval():
    pass

def predict_forecast():
    pass

def detect_anomaly():
    pass

if __name__ == "__main__":
    launcher(data_file="AirQualityUCIv", time_index="DateTime", targets=[
        "CO(GT)","PT08.S1(CO)","NMHC(GT)","C6H6(GT)","PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)","NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH","AH"
    ])
    # launcher(data_file="energydata_complete", time_index="date", targets=["Appliances"])

    
    # root_dir = os.path.dirname(os.getcwd())
    # data_dir = os.path.join(root_dir, "data/")
    # data_list = os.listdir(data_dir)
    
    # for data_file in data_list:
    #     if 'train' in data_file or 'test' in data_file:
    #         continue
        
    #     filepath = os.path.join(data_dir, data_file)
    #     if os.path.isfile(filepath):
    #         print(">>>>>", data_file[:-4], "<<<<<")
    #         launcher(data_file[:-4])