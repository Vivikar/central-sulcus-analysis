from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm import tqdm
from data.bvisa_dm import CS_DataModule
from models.UNet3D import UNet3D

import yaml
import torch 

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    
    return result

torch.set_float32_matmul_precision('medium')

def main():
    # # read the config file
    # with open('config.yaml', 'r') as f:
    #     cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # saves top-K checkpoints based on "valid_dsc" metric
    # checkpoint_callback = ModelCheckpoint(save_top_k=5,
    #                                       monitor="valid_dsc_macro_epoch",
    #                                       mode="max",
    #                                       filename="{epoch:02d}-{valid_dsc_macro_epoch:.4f}")
    # # enable early stopping (NOT USED RN)
    # early_stop_callback = EarlyStopping(monitor="valid_dsc_macro_epoch",
    #                                     min_delta=0.0001,
    #                                     patience=10,
    #                                     verbose=False,
    #                                     mode="max")
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print('beg', get_gpu_memory_map())
    cfg = {'exp_name': 'test'}

    
    # prepare data
    data_module = CS_DataModule()
    # data_module = data_module.data
    data_module.prepare_data()

    # get model and trainer
    model = UNet3D()

    traindl = data_module.train_dataloader()
    for batch in tqdm(traindl):
        break
    image = batch['image']
    print('img shape', image.shape)

    forward = model(image)
    print('forw shape', forward.shape)
    print('after forward', get_gpu_memory_map())

if __name__ == '__main__':
    main()