from data_loader import PLDataLoader
from lstm_model import VanillaLSTMModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    # training settings
    hparams = {
        "input_size": 1,
        "hidden_size": 1,
        "output_size": 1,
        "lr": 0.01,
    }
    # instantiate model
    model = VanillaLSTMModule(hparams)

    # data loaders
    loader = PLDataLoader()
    loader.setup()

    logger = TensorBoardLogger('logs', name='vanilla_lstm')
    # train model
    trainer = Trainer(logger, max_epochs=10)
    trainer.fit(model, loader)

if __name__ == '__main__':
    main()
