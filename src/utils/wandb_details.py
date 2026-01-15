from dataclasses import dataclass


@dataclass
class WandbDetails:
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str | None = None
    init_project: bool = True
