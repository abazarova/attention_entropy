import os

import hydra
import yaml
from comet_ml import Experiment
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from src.evaluation import evaluate
from src.evaluation.process_metrics import process_metrics
from src.methods.hallucination_detection_abc import HallucinationDetectionMethod, T
from src.preprocess.dataset_abc import HallucinationDetectionDataset

load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="unsupervised")
def main(cfg: OmegaConf):
    hydra_cfg = HydraConfig.get()
    preprocess_name: str = hydra_cfg.runtime.choices["preprocess"]
    method_name: str = hydra_cfg.runtime.choices["method"]
    model_name: str = cfg["model_name"]
    transfer_names: list[str] = cfg["transfer_names"]
    experiment_name = f"{preprocess_name}_{method_name}_{model_name}"

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="llm-factuality",
    )

    # Set experiment name
    experiment.set_name(experiment_name)
    experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    dataset: HallucinationDetectionDataset = instantiate(cfg["preprocess"])
    X, y, _, _ = dataset.process()

    model: HallucinationDetectionMethod = instantiate(cfg["method"], _convert_="all")
    assert method_name not in [
        "mtopdiv",
        "linear_probe",
        "linear_probe_simple",
        "lookback",
    ], f"This method is not supported: {method_name}"

    metrics, best_model = evaluate(
        model,
        X,
        y,
        **cfg["evaluation"],
    )

    table_str, raw_table = process_metrics(metrics, experiment)
    logger.success(f"Results for cross validation on {dataset.__class__.__name__}")
    print(table_str)
    experiment.log_metric("final_test_auroc", raw_table["roc_auc"].loc["test"]["mean"])

    logger.info("Transfering model on another dataset")
    for transfer_name in transfer_names:
        with open(f"config/transfer/{transfer_name}.yaml") as f:
            transfer_cfg = yaml.load(f, Loader=yaml.FullLoader)
        transfer_cfg["model_name"] = model_name
        transfer_dataset: HallucinationDetectionDataset = instantiate(transfer_cfg)
        X, y, _, _ = transfer_dataset.process()

        metrics, _ = evaluate(
            best_model,
            X,
            y,
            tune_hyperparameters=False,
            pretrained=True,
            save_best_model=False,
            **cfg["evaluation"],
        )

        table_str, raw_table = process_metrics(metrics, experiment)
        logger.success(
            f"Transfer for {transfer_name} on {transfer_dataset.model_name} model"
        )
        experiment.log_metric(
            f"{transfer_name}_transfer_roc_auc",
            raw_table["roc_auc"].loc["test"]["mean"],
        )
        print(table_str)

    experiment.end()


if __name__ == "__main__":
    main()
