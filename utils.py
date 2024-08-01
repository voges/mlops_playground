import logging
import requests

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
    log.info("Configuration:")
    log.info("--------------")
    yaml_str = OmegaConf.to_yaml(cfg=cfg)
    for line in yaml_str.splitlines():
        log.info(f"  {line}")


def initialize_mlflow_logger(
    cfg: DictConfig, experiment_name: str, run_name: str
) -> MLFlowLogger:
    mlf_logger = None
    if cfg.mlflow.enabled:
        if not check_server(uri=cfg.mlflow.tracking_uri):
            log.error("MLflow tracking server not available.")
        else:
            log.info(f"MLflow experiment name: {experiment_name}")
            log.info(f"MLflow run name: {run_name}")

            mlf_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=cfg.mlflow.tracking_uri,
            )
    return mlf_logger
