
import torch as t
import pytorch_lightning as pl
import ipdb

class VanillaLSTM(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = t.nn.LSTM(input_size, hidden_size)
        self.act = t.nn.Tanh()
        # self.linear = t.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        # tanh on output
        output = self.act(output)
        return output 

    def init_hidden(self):
        return t.autograd.Variable(t.zeros(1, 1, self.hidden_size))




# VanillaLSTM pytorch lightning module
class VanillaLSTMModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.params = hparams
        # ipdb.set_trace()
        self.lstm = VanillaLSTM(
            input_size=hparams['input_size'],
            hidden_size= hparams['hidden_size'],
            output_size= hparams['output_size'],
        )
        self.loss_fn = t.nn.MSELoss()

    def forward(self, x):
        return self.lstm(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.params['lr'])

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

    def on_epoch_end(self):
        pass

    def on_train_end(self):
        pass
