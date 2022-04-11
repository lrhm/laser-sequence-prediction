from data_loader import PLDataLoader
from lstm_model import VanillaLSTMModule
from pytorch_lightning import Trainer


# training settings
hparams = {
    "input_size": 1,
    "hidden_size": 32,
    "output_size": 1,
    "lr": 0.01,
}


# instantiate model
model = VanillaLSTMModule(hparams)

# data loaders
loader = PLDataLoader()
loader.setup()

# train model
trainer = Trainer(max_epochs=10)
trainer.fit(model, loader)