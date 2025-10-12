import pathlib

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from src.constants import HYDRA_CONFIG_PATH, PROJECT_PATH

## Define own callbacks here
def _resolve_checkpoint_root(cfg) -> pathlib.Path:
    """
    try to use cfg.paths.checkpoint_root first, if not, fallback to PROJECT_PATH/checkpoints
    """
    checkpoint_root = getattr(getattr(cfg, "paths", None), "checkpoint_root", None)
    if checkpoint_root:
        return pathlib.Path(checkpoint_root).expanduser().resolve()
    return (PROJECT_PATH / "checkpoints").resolve()



def get_callbacks(cfg):
    callacks = []
    # ckpt_dir = pathlib.Path('/p/project/hai_denovo/Projects/debug/misato-ba/configs/runs')
    # ckpt_dir = pathlib.Path(HYDRA_CONFIG_PATH).joinpath("runs", cfg.name)
    ckpt_dir = _resolve_checkpoint_root(cfg) / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if "checkpointing" in cfg.callbacks:
        ckpt_dir.mkdir(exist_ok=True)
        # Saves the top k checkpoints according to the test metric throughout
        # training.
        monitor_metric = f"{cfg.trainer.eval_metrics}"
        monitor_mode = "max" if "f1" in monitor_metric.lower() else "min"
        print(f"monitor_metric: {monitor_metric}")
        print(f"monitor_mode: {monitor_mode}")
        # Clean metric name for filename
        metric_name = monitor_metric.replace("/", "_")
        checkpoint_filename = f"{cfg.name}_{{epoch}}-{{{metric_name}:.4f}}"
        print(f"checkpoint_filename: {checkpoint_filename}")
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=checkpoint_filename,#f"{cfg.name}" + "_{epoch}-{val_loss:.4f}",  # {epoch}-{val_loss:.4f}",
            every_n_epochs=cfg.callbacks.checkpointing.checkpoint_freq,
            monitor=monitor_metric, #f"{cfg.trainer.eval_metrics}",
            save_top_k=cfg.callbacks.checkpointing.save_top_k,
            mode=monitor_mode #"min", #min 
        
            # save_last=True,  # Add this line to also save last epoch
        )
        callacks.append(ckpt)

        # Save last epoch checkpoint
        last_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=checkpoint_filename + "_last",#f"{cfg.name}" + "_{epoch}-{val_loss:.4f}" + "_last",
            save_last=True,
        )
        callacks.append(last_ckpt)

    # TODO add your own callbacks here

    return callacks
