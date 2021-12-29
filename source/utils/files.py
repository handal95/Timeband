import os
import pickle

def get_path(dirname: str, filename: str, postfix: str = "") -> os.path:
    filename = filename if postfix == "" else f"{filename}_{postfix}"
    filepath = os.path.join(dirname, f"{filename}.pkl")
    return filepath

def load_core(core_path:os.path):
    with open(core_path, "rb") as f:
        core = pickle.load(f)

    return core

def save_core(core: "Timeband", core_path: os.path, best: bool = False):
    if best:
        print(f"Best Model is Saved at {core_path}")

    with open(core_path, mode="wb") as f:
        pickle.dump(core, f)
    