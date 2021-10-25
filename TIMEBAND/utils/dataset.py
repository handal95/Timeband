import torch

class Dataset:
    def __init__(self, encoded, decoded, missing):
        self.encoded = encoded
        self.decoded = decoded
        self.missing = missing

        self.length = len(self.encoded)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {
            "encoded": torch.tensor(self.encoded[idx], dtype=torch.float32),
            "decoded": torch.tensor(self.decoded[idx], dtype=torch.float32),
            "missing": torch.tensor(self.missing[idx], dtype=torch.float32),
            # "anomals": torch.tensor(self.decoded[idx], dtype=torch.float32),
        }

        return data