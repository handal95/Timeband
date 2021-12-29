import os
import datetime
import argparse

class ArgParser:
    def __init__(self, data_file):
        parser = self.get_parser()
        args = parser.parse_args()
        args.data_file = data_file

        self.args = self.check_constraints(args)
    
    def get_parser(self):
        parser = argparse.ArgumentParser(description='** TIMEBAND CLI Parser. **')
        
        # Directory path option
        parser.add_argument("-dd", "--data_dir", type=str, help="Dir name of datasets", default="data/")
        parser.add_argument("-ld", "--logs_dir", type=str, help="Dir name of log files", default="logs/")
        parser.add_argument("-md", "--model_dir", type=str, help="Dir name of pretrained models", default="models/")
        
        # File name option
        parser.add_argument("-df", "--data_file", type=str, help="File name of dataset")
        parser.add_argument("-mf", "--model_file", type=str, help="File name of pretrained model")

        # Logging option
        parser.add_argument("-l", "--logging", type=str2bool, help="Flag of logging", default=True)
        parser.add_argument("-f", "--logfile", type=str, help="Name of log file", default=None)
        parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level of log", default=1)
        
        # Launcher mode option
        parser.add_argument("-t", "--train", type=str2bool, help="Flag of train option")
        parser.add_argument("-p", "--predict", type=str2bool, help="Flag of predict option")
        parser.add_argument("-d", "--detect", type=str2bool, help="Flag of anomaly detect option")
        parser.set_defaults(train=True, predict=True, detect=True)

        # Model option
        parser.add_argument("-b", "--band_width", type=int, help="Bandwitdth of normal values")
        parser.add_argument("-e", "--epochs", type=int, help="Training Epochs")
        parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")

        parser.add_argument("-T", "--targets", type=list, help="Target columns for prediction")
        parser.add_argument("-l1", "--l1_weight", type=int, help="Weight for L1 loss", default=2e-3)
        parser.add_argument("-l2", "--l2_weight", type=int, help="Weight for L2 loss", default=2e-3)
        parser.add_argument("-gp", "--gp_weight", type=int, help="Weight for gradients penalty", default=2e-4)
        
        # Dataset option
        parser.add_argument("-ti", "--time_index", type=str, help="Timestamps index column")
        parser.add_argument("-ow", "--observed_length", type=int, help="Length of observed window")
        parser.add_argument("-fw", "--forecast_length", type=int, help="Length of forecast window")
        parser.set_defaults(time_index="timestamp", observed_length=6, forecast_length=3)
        
        # Time encode option
        parser.add_argument("-ty", "--year", type=str2bool, help="Encoding Flag of YEAR feature")
        parser.add_argument("-tm", "--month", type=str2bool, help="Encoding Flag of MONTH feature")
        parser.add_argument("-tw", "--dayofweek", type=str2bool, help="Encoding Flag of DAYOFWEEK feature")
        parser.add_argument("-td", "--day", type=str2bool, help="Encoding Flag of DAY feature")
        parser.add_argument("-th", "--hour", type=str2bool, help="Encoding Flag of HOUR feature")
        parser.set_defaults(year=False, month=False, dayofweek=False, day=False, hour=False)

        return parser
    
    def check_constraints(self, args):
        # Data Files
        base_path = os.path.dirname(os.getcwd())
        data_path = os.path.join(base_path, args.data_dir, args.data_file)
        if not os.path.exists(data_path) and not os.path.exists(f"{data_path}.csv"):
            raise FileNotFoundError(f"{data_path} is NOT EXISTS")
        
        args.model_file = args.model_file if args.model_file is not None else args.data_file
    
        # Log Files
        if args.logging is True:
            if args.logfile is None:
                today = datetime.datetime.today()
                args.logfile = (today).strftime("%Y-%m-%d")
        else:
            args.logfile = None
            
        # Dir Path
        args.data_dir = os.path.join(base_path, args.data_dir)
        args.logs_dir = os.path.join(base_path, args.logs_dir)
        args.model_dir = os.path.join(base_path, args.model_dir)

        return args
            
            
def str2bool(value: str):
    if isinstance(value, bool):
        return value

    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")