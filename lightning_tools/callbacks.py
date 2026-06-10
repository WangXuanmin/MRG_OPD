import os
# from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch import loggers as pl_loggers
# from lightning.pytorch.callbacks import LearningRateMonitor
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)
    every_n_train_steps = args.every_n_train_steps or None

    # --------- Add Callbacks
    if args.model_name == 'branchgpt':
        callbacks = []
        if args.early_stop:
            callbacks.append(EarlyStopping(
                monitor="val_score",
                mode="max",
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                verbose=True,
            ))
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="{epoch}-{step}",
            save_top_k=-1,
            every_n_train_steps=every_n_train_steps,
            save_last=False,
            save_weights_only=False
        )
        callbacks = [checkpoint_callback]
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": callbacks + [lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns
