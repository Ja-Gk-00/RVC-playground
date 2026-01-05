from dataclasses import dataclass


@dataclass
class WandbDetails:
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str | None = None
    # Set init_project to False to manually call wandb.init()/wandb.finish() to be able to call train() multiple times in a single experiment
    init_project: bool = True
