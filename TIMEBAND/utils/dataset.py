import torch


class Dataset:
    def __init__(self, encoded, decoded):
        self.encoded = encoded
        self.decoded = decoded

        self.length = len(self.encoded)

    def __len__(self):
        return self.length

    def shape(self, target="encode"):
        return self.encoded.shape if target == "encode" else self.decoded.shape

    def __getitem__(self, idx):
        data = {
            "encoded": torch.tensor(self.encoded[idx], dtype=torch.float32),
            "decoded": torch.tensor(self.decoded[idx], dtype=torch.float32),
        }

        return data
