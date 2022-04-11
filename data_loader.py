from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch as t
from torch.utils.data import TensorDataset, DataLoader
import torch as t
import ipdb
from tqdm import tqdm


class PLDataLoader(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./norm_data.pt",
        batch_size: int = 32,
        seq_size: int = 10,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        raw_data = t.load(self.data_dir)
        frames = tuple(
            (raw_data[i : i + self.seq_size], raw_data[i + self.seq_size + 1])
            for i in range(len(raw_data) - self.seq_size - 1)
        )
        xs = t.stack(tuple(frame[0] for frame in frames))
        ys = t.stack(tuple(frame[1] for frame in frames))
        train_xs = xs[: int(t.floor(t.tensor(xs.shape[0] * 0.8)))]
        train_ys = ys[: int(t.floor(t.tensor(xs.shape[0] * 0.8)))]
        test_xs = xs[int(t.floor(t.tensor(xs.shape[0] * 0.2))) :]
        test_ys = ys[int(t.floor(t.tensor(xs.shape[0] * 0.2))) :]
        self.train_dataset = TensorDataset(train_xs, train_ys)
        self.test_dataset = TensorDataset(test_xs, test_ys)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def main():
    loader = PLDataLoader()
    loader.setup()
    train = loader.train_dataloader()
    for x, y in tqdm(train):
        ipdb.set_trace()
        pass


if __name__ == "__main__":
    main()
