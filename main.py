import logging

import lightning as L

from omegaconf import DictConfig
import hydra

import optuna

import data
import models
from utils import log_config, initialize_mlflow_logger

log = logging.getLogger(__name__)


def train_and_test_model(cfg: DictConfig, latent_dim: int, mlf_logger=None):
    # Load the data and create data loaders.
    train_set, test_set = data.load_data()
    train_loader, test_loader = data.create_data_loaders(
        cfg=cfg, train_set=train_set, test_set=test_set
    )

    # Create an instance of the autoencoder model.
    autoencoder = models.AutoEncoder(lr=cfg.train.learning_rate, latent_dim=latent_dim)

    # Create a trainer and train the autoencoder.
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=mlf_logger,
        default_root_dir=cfg.lightning.checkpoint_path,
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Test the model.
    trainer.test(model=autoencoder, dataloaders=test_loader)
    test_loss = trainer.callback_metrics["test_loss"]

    return test_loss


class Objective:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, trial: optuna.Trial):
        optuna_study_name = trial.study.study_name
        optuna_trial_number = trial.number

        log.info(f"Optuna study name: {optuna_study_name}")
        log.info(f"Optuna trial number: {optuna_trial_number}")

        mlf_logger = initialize_mlflow_logger(
            cfg=self.cfg,
            experiment_name=optuna_study_name,
            run_name=f"{optuna_trial_number:04d}",
        )

        latent_dim = trial.suggest_int("latent_dim", low=2, high=8)

        test_loss = train_and_test_model(
            cfg=self.cfg, latent_dim=latent_dim, mlf_logger=mlf_logger
        )

        return test_loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    log_config(cfg=cfg)

    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction="minimize",
        storage=cfg.optuna.db_path,
        load_if_exists=True,
    )

    study.optimize(func=Objective(cfg=cfg), n_trials=cfg.optuna.n_trials)


if __name__ == "__main__":
    main()