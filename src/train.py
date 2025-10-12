"""
Main module to load and train the model. This should be the program entry point.
"""
import warnings

import hydra
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import sys
import os
# remove proxy environment variables
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Add the project directory to the sys.path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, '..'))

from src import constants
from src.utils.callbacks import get_callbacks
from src.utils.logutils import get_lightning_logger, get_logger
from omegaconf import OmegaConf
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Logger initialization
logger = get_logger(__name__)


@hydra.main(
    config_path=constants.HYDRA_CONFIG_PATH,
    config_name=constants.HYDRA_CONFIG_NAME,
    version_base=constants.HYDRA_VERSION_BASE,
)
def train(config: DictConfig):
    """
    Train model with PyTorch Lightning and log with Wandb.

    Parameters:
    - config: A configuration object with the training parameters.
    """
    # Set random seeds
    seed_everything(config.seed)
    logger.info("Set random seed to %d", config.seed)
    torch.set_float32_matmul_precision("high")

    # Load data module
    # datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule = hydra.utils.instantiate(config.datamodule, graph_type=config.graph_type)
    datamodule.setup()

    # Validate the loaded data
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader =  datamodule.test_dataloader()
    logger.info("Loaded datamodule")
    logger.info(f"Number of training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Number of test samples: {len(test_dataloader.dataset)}")
    # Load the model
    logger.info("Loading model")
    model = hydra.utils.instantiate(config.model)
    print(model)
    num_params = []
    num_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Number of parameters: {num_params}")
    # Setup logging and checkpointing
    # pl_logger = get_lightning_logger(config)
    # Setup logging and checkpointing
    if config.name == "test":
        wandb_logger = None
    elif config.enable_wandb:
        wandb_logger = WandbLogger(
            project=config.logger.project,
            entity=config.logger.entity,
            name=config.name,
            offline=False,
            config=OmegaConf.to_container(config, resolve=True, enum_to_str=True)  # 只传递基本数据类型
        )
    else:
        wandb_logger = None

    callbacks = get_callbacks(config)

    # Instantiate and start the Trainer
    logger.info(f"batch_size: {config.datamodule.batch_size}, accumulate_grad_batches: {config.trainer.accumulate_grad_batches}")
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=config.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=config.trainer.log_steps,
        val_check_interval=config.trainer.val_interval,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=config.trainer.gradient_clip_val,
    )
    trainer.fit(model, datamodule)

    # Test the model at the best checkpoint
    #if config.test_model:
    #    logger.info("Testing the model at the best checkpoint")
    #    trainer.test(ckpt_path="best")
    #    logger.info("Train loop completed. Exiting.")


if __name__ == "__main__":
    train()
