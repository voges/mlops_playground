import logging
import requests
import subprocess
from typing import Optional

import torch

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import MLFlowLogger

log = logging.getLogger(__name__)


def check_server(uri: str, timeout: int = 5) -> bool:
    """Check if the server is available."""
    try:
        response = requests.get(url=uri, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            log.warning(f"Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"Error connecting to server: {e}")
        return False


def log_config(cfg: DictConfig) -> None:
    log.info("")
    log.info("Configuration:")
    log.info("--------------")
    yaml_str = OmegaConf.to_yaml(cfg=cfg)
    for line in yaml_str.splitlines():
        log.info(f"  {line}")
    log.info("")


def initialize_mlflow_logger(
    cfg: DictConfig, experiment_name: str, run_name: str
) -> MLFlowLogger:
    if not cfg.mlflow.enabled:
        log.info("MLflow is not enabled in the configuration.")
        return None

    if not check_server(uri=cfg.mlflow.tracking_uri):
        log.error("MLflow tracking server not available.")
        return None

    log.info(f"MLflow experiment name: {experiment_name}")
    log.info(f"MLflow run name: {run_name}")

    try:
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=cfg.mlflow.tracking_uri,
        )
        return mlf_logger
    except Exception as e:
        log.error(f"Failed to initialize MLFlowLogger: {e}")
        return None


def identify_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_git_root() -> Optional[str]:
    try:
        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        return git_root
    except subprocess.CalledProcessError:
        return None
