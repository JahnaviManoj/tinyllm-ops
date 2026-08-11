"""QLoRA fine-tune (tutorial 2.7): frozen 4-bit NF4 base + trainable LoRA
adapters on the attention layers, via TRL's SFTTrainer + PEFT.

Zero hardcoded hyperparameters — everything comes from the YAML config (2.6),
so "reproduce run N" literally means --config configs/exp_N.yaml. Data arrives
only through hash manifests (fetched + verified if the local copy is missing
or stale), the prompt format comes from tinyllm.prompt (the ONE place it may
live), and every run logs params, git commit, loss curves and the merged
model to MLflow (local mlruns until 2.8 re-points at Azure ML).

Two hardware notes learned the hard way in 2.5:
  - Gemma overflows fp16 (NaN logits, all-pad output). Compute dtype is bf16
    where supported, else fp32. fp16 is never used.
  - A 4GB GTX 1650 Ti fits this fine: NF4 base ~0.2GB + LoRA + gradient
    checkpointing. No Colab required at 270M.

CLI: uv run python -m tinyllm.train --config configs/exp_001.yaml
     [--max-steps N --limit N]   (smoke runs)
"""

import argparse
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch

from tinyllm.config import ExperimentConfig, load_config
from tinyllm.manifest import fetch_dataset
from tinyllm.prompt import format_example


def set_seed(seed: int):  # same seed → same run → reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_from_manifest(
    manifest_path: str,
    limit: int | None = None,
    eos: str = "",
    sender_id_frac: float = 0.0,
    seed: int = 42,
):
    """Dataset by content hash, never by filename trust: verify the local
    file against the manifest and re-fetch from Blob on any mismatch.

    `eos` is appended to every text — without it the model never sees a stop
    signal (the first sweep's outputs trailed junk; found in the 2.9 autopsy).
    `sender_id_frac` applies the DLT sender-ID augmentation to a seeded
    fraction of rows at load time (train split only — val stays clean so val
    loss remains comparable across configs)."""
    from datasets import Dataset

    from tinyllm.augment import add_sender_id

    m = json.load(open(manifest_path))
    local = os.path.join("data/generated", os.path.basename(m["blob_path"]))
    ok = (
        os.path.exists(local)
        and hashlib.sha256(open(local, "rb").read()).hexdigest() == m["sha256"]
    )
    if not ok:
        fetch_dataset(manifest_path, local)
    rows = [json.loads(line) for line in open(local)][:limit]
    rng = random.Random(seed)
    texts = []
    for r in rows:
        if sender_id_frac and rng.random() < sender_id_frac:
            r = {**r, "sms": add_sender_id(r["sms"], rng)}
        texts.append({"text": format_example(r)["text"] + eos})
    return Dataset.from_list(texts)


def flat_params(cfg: ExperimentConfig) -> dict:
    out = {}
    for k, v in cfg.model_dump().items():
        if isinstance(v, dict):
            out.update({f"{k}.{k2}": v2 for k2, v2 in v.items()})
        else:
            out[k] = v
    return out


def train(
    config_path: str,
    max_steps: int | None = None,
    limit: int | None = None,
    log_merged: bool = False,
    resume_from: str | None = None,
):
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    import mlflow

    cfg = load_config(config_path)
    set_seed(cfg.seed)
    # Per-config output dirs: sweep runs must not overwrite each other.
    out_root = os.path.join(
        "outputs", os.path.splitext(os.path.basename(config_path))[0]
    )
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    mlflow.set_experiment("tinyllm-finetune")  # local mlruns until 2.8
    with mlflow.start_run(run_name=os.path.basename(config_path)):
        mlflow.log_params(flat_params(cfg))
        mlflow.log_param("config_path", config_path)
        mlflow.log_param(
            "git_commit",
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        )
        mlflow.log_param(
            "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        mlflow.log_param("compute_dtype", str(compute_dtype))

        bnb = BitsAndBytesConfig(  # the "Q" in QLoRA
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tok = AutoTokenizer.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, quantization_config=bnb
        )

        peft_cfg = LoraConfig(  # the "LoRA"
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )

        frac = cfg.augment.sender_id_frac if cfg.augment else 0.0
        val = (
            load_from_manifest(cfg.val_manifest, limit, eos=tok.eos_token)
            if cfg.val_manifest
            else None
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=load_from_manifest(
                cfg.dataset_manifest,
                limit,
                eos=tok.eos_token,
                sender_id_frac=frac,
                seed=cfg.seed,
            ),
            eval_dataset=val,
            peft_config=peft_cfg,
            args=SFTConfig(
                output_dir=os.path.join(out_root, "run"),
                learning_rate=cfg.train.learning_rate,
                num_train_epochs=cfg.train.epochs,
                max_steps=max_steps if max_steps else -1,
                per_device_train_batch_size=cfg.train.batch_size,
                gradient_accumulation_steps=cfg.train.gradient_accumulation,
                gradient_checkpointing=False,  # 270M + short SMS fit 4GB without it; ~2x faster
                max_length=512,
                eval_strategy="epoch" if val is not None else "no",
                bf16=compute_dtype is torch.bfloat16,
                fp16=False,  # Gemma + fp16 = NaN (see 2.5)
                seed=cfg.seed,
                logging_steps=10,
                report_to="mlflow",
            ),
        )
        if resume_from:  # restores weights + optimizer + scheduler + RNG state
            mlflow.log_param("resumed_from_checkpoint", resume_from)
        trainer.train(resume_from_checkpoint=resume_from)
        adapter_dir = os.path.join(out_root, "adapter")
        trainer.save_model(adapter_dir)
        mlflow.log_artifacts(adapter_dir, artifact_path="adapter")  # small, always

        # Merge adapters back into the base → one normal model (GGUF needs this).
        # Re-merge against a CLEAN fp32 base on CPU: merging into the 4-bit
        # quantized weights would bake quantization error into the artifact.
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
        merged_dir = os.path.join(out_root, "merged")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.to(torch.bfloat16 if compute_dtype is torch.bfloat16 else torch.float32)
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)  # GGUF conversion needs the tokenizer
        if log_merged:  # ~1.1GB — sweep runs skip this; the winner logs it
            mlflow.log_artifacts(merged_dir, artifact_path="model")
        print(
            f"merged model + tokenizer → {merged_dir}"
            + (" (logged to MLflow)" if log_merged else "")
        )


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune from a YAML experiment config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="cap steps (smoke runs)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap dataset rows (smoke runs)"
    )
    parser.add_argument(
        "--log-merged",
        action="store_true",
        help="upload the merged model to MLflow (winner runs only)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="trainer checkpoint dir, e.g. outputs/exp_010/run/checkpoint-500",
    )
    args = parser.parse_args()
    train(
        args.config,
        max_steps=args.max_steps,
        limit=args.limit,
        log_merged=args.log_merged,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
