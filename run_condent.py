import os
from itertools import product
from pathlib import Path

import hydra
import pandas as pd
import yaml
from comet_ml import Experiment
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from src.evaluation import evaluate
from src.evaluation.process_metrics import process_metrics
from src.preprocess.dataset_abc import HallucinationDetectionDataset
from src.methods.condent import CondEntRAUQ, CondEnt

load_dotenv()


@hydra.main(version_base=None, config_path="config", config_name="condent")
def main(cfg: OmegaConf):
    hydra_cfg = HydraConfig.get()
    preprocess_name: str = hydra_cfg.runtime.choices["preprocess"]
    method_name: str = hydra_cfg.runtime.choices["method"]
    model_name: str = cfg["model_name"]
    transfer_names: list[str] = cfg["transfer_names"]
    experiment_name = f"{preprocess_name}_{method_name}_{model_name}"

    experiment = None
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="llm-factuality",
    )

    # Set experiment name
    experiment.set_name(experiment_name)
    experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    dataset: HallucinationDetectionDataset = instantiate(cfg["preprocess"])
    X, y, _, _ = dataset.process()

    model: CondEnt | CondEntRAUQ = instantiate(cfg["method"], _convert_="all")

    assert method_name in ["condent", "condent_rauq"], f"This method is not supported: {method_name}"

    tune_hyperparameters = model.analysis_sites == "all"

    if model.analysis_sites == "all":
        model.analysis_sites = sorted(
            product(range(model.n_layers), range(model.n_heads))
        )

    metrics, best_model = evaluate(
        model,
        X,
        y,
        tune_hyperparameters=tune_hyperparameters,
        **cfg["evaluation"],
    )


    table_str, raw_table = process_metrics(metrics, experiment)
    logger.success(f"Results for cross validation on {dataset.__class__.__name__}")
    print(table_str)
    experiment.log_metric(
        "final_test_auroc", raw_table["roc_auc"].loc["test"]["mean"]
    )
    
    # Save results to CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_filename = f"results/{model_name}_{preprocess_name}.csv"
    
    # Extract the required values
    alpha_value = cfg["method"]["alpha"] if "alpha" in cfg["method"] else None
    aggregation_value = cfg["method"]["aggregation"] if "aggregation" in cfg["method"] else None
    
    val_mean = raw_table["roc_auc"].loc["val"]["mean"]
    test_mean = raw_table["roc_auc"].loc["test"]["mean"]
    
    # Create results row
    results_row = {
        "preprocess_name": preprocess_name,
        "method_alpha": alpha_value,
        "method_aggregation": aggregation_value,
        "val_mean": val_mean,
        "test_mean": test_mean
    }
    
    # Save to CSV
    if os.path.exists(csv_filename):
        # Append to existing file
        existing_df = pd.read_csv(csv_filename)
        new_df = pd.DataFrame([results_row])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(csv_filename, index=False)
    else:
        # Create new file
        pd.DataFrame([results_row]).to_csv(csv_filename, index=False)
    
    logger.info(f"Results saved to {csv_filename}")

    logger.info("Transfering model on another dataset")
    for transfer_name in transfer_names:
        if preprocess_name != transfer_name:
            with open(f"config/transfer/{transfer_name}.yaml") as f:
                transfer_cfg = yaml.load(f, Loader=yaml.FullLoader)
            transfer_cfg["model_name"] = model_name
            transfer_dataset: HallucinationDetectionDataset = instantiate(transfer_cfg)
            X, y, _, _ = transfer_dataset.process()
            model.cache_dir = (
                Path(cfg["method"]["cache_dir"]).parent
                / transfer_name
                / f"zero_out_{cfg['method']['zero_out']}"
                / model_name
            )

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
            if cfg["log_output"]:
                experiment.log_metric(
                    f"{transfer_name}_roc_auc", raw_table["roc_auc"].loc["test"]["mean"]
                )
            print(table_str)
            
    experiment.end()


if __name__ == "__main__":
    main()
