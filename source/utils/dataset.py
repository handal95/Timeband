import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, encoded, decoded):
        super(Dataset, self).__init__()
        self.encoded = encoded
        self.decoded = decoded

    def shape(self, target="encode"):
        return self.encoded.shape if target == "encode" else self.decoded.shape

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        data = {
            "encoded": torch.tensor(self.encoded[idx], dtype=torch.float32),
            "decoded": torch.tensor(self.decoded[idx], dtype=torch.float32),
        }

        return data
