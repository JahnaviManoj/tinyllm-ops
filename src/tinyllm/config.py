# YAML in, validated object out (typos crash immediately)
import yaml
from pydantic import BaseModel


class LoraConfig_(BaseModel):
    r: int
    alpha: int
    target_modules: list[str]


class TrainParams(BaseModel):
    learning_rate: float
    epochs: int
    batch_size: int
    gradient_accumulation: int


class AugmentParams(BaseModel):
    sender_id_frac: float = 0.0  # fraction of train SMS given a DLT sender-ID prefix


class ExperimentConfig(BaseModel):
    model_id: str
    dataset_manifest: str
    val_manifest: str | None = None  # eval loss per epoch when set
    seed: int
    lora: LoraConfig_
    train: TrainParams
    augment: AugmentParams | None = None


def load_config(path: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(open(path)))
