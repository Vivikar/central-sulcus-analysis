from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.utils.default as utils
import matplotlib.pyplot as plt


log = utils.get_pylogger(__name__)

EVAL_PATH = '../logs/test_train/runs/2023-02-17_13-03-42/.hydra'
CHKPT_PATH = '/mrhome/vladyslavz/git/central-sulcus-analysis/logs/test_train/runs/2023-02-17_13-03-42/checkpoints/epoch_001.ckpt'


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    cfg.ckpt_path = CHKPT_PATH
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    for batch in datamodule.val_dataloader():
        image = batch['image']
        target = batch['target']
    plt.imshow(image[0, 0, 80, :, :])
    plt.imshow(target[0, 0, 80, :, :])

    
    
@hydra.main(version_base="1.3", config_path=EVAL_PATH, config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)
if __name__ == "__main__":
    main()