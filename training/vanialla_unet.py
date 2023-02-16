from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm import tqdm
from data.dataset import CS_DataModule
from models.UNet3D import UNet3D

import yaml
import torch 

torch.set_float32_matmul_precision('medium')

def main():
    # # read the config file
    # with open('config.yaml', 'r') as f:
    #     cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # saves top-K checkpoints based on "valid_dsc" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          monitor="validation_dsc",
                                          mode="max",
                                          filename="{epoch:02d}-{validation_dsc:.4f}")
    # # enable early stopping (NOT USED RN)
    # early_stop_callback = EarlyStopping(monitor="valid_dsc_macro_epoch",
    #                                     min_delta=0.0001,
    #                                     patience=10,
    #                                     verbose=False,
    #                                     mode="max")

    cfg = {'exp_name': 'test'}
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=Path('./outputs'),
                               name=cfg['exp_name'],
                               version=0)
    # prepare data
    data_module = CS_DataModule()
    # data_module = data_module.data
    data_module.prepare_data()
    
    # get model and trainer
    model = UNet3D()
    
    # save the config file to the output folder
    # # for a given experiment
    # dump_path = Path('./outputs').resolve() / f'{cfg["exp_name"]}'
    # dump_path.mkdir(parents=True, exist_ok=True)
    # dump_path = dump_path/'config_dump.yml'
    # with open(dump_path, 'w') as f:
    #     yaml.dump(cfg, f)
    
    
    trainer = Trainer(max_epochs=2,
                      log_every_n_steps=1,
                      accelerator='gpu',
                      logger=logger,
                      auto_lr_find=True,
                      callbacks=[checkpoint_callback],
                      )

    # # # find optimal learning rate
    # print('Default LR: ', model.learning_rate)
    # trainer.tune(model, datamodule=data_module)
    # print('Tuned LR: ', model.learning_rate)

    # train model
    print("Training model...")
    trainer.fit(model=model,
                datamodule=data_module)
    
if __name__ == "__main__":
    main()