# YAML in, validated object out (typos crash immediately)
from typing import Literal

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
    model_revision: str | None = None  # pin the exact HF commit (era-2 rule)
    chat_format: bool = (
        False  # True for instruct models (Qwen); False = era-1 raw prompt
    )
    dataset_manifest: str
    val_manifest: str | None = None  # eval loss per epoch when set
    seed: int
    lora: LoraConfig_
    train: TrainParams
    augment: AugmentParams | None = None
    gradient_checkpointing: bool = False  # ON for 2B on T4; OFF for 270M (speed)
    max_length: int = 640  # chat scaffolding adds ~40 tokens over era-1's 512
    # auto = bf16 on Ampere+ else fp32. T4 / GTX 16xx: set fp16 explicitly after
    # scripts/gen_smoke.py passes (§2.2). A typo here fails at load, not mid-run.
    compute_dtype: Literal["auto", "fp16", "fp32"] = "auto"


def load_config(path: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(open(path)))
