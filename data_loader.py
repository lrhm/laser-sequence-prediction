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

    def setup(self):
        raw_data = t.load(self.data_dir)
        frames = t.stack(
            tuple(
                raw_data[i : i + self.seq_size]
                for i in range(len(raw_data) - self.seq_size)
            )
        )
        frames = frames[t.randperm(len(frames))]
        train_data = frames[: int(t.floor(t.tensor(frames.shape[0] * 0.8)))]
        test_data = frames[int(t.floor(t.tensor(frames.shape[0] * 0.2))) :]
        self.train_dataset = TensorDataset(train_data)
        self.test_dataset = TensorDataset(test_data)
        """
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000]
        )
        """

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
    for d in tqdm(train):
        pass


if __name__ == "__main__":
    main()
