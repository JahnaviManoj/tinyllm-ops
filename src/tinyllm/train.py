"""QLoRA fine-tune (tutorial §2.3, Era 2): frozen 4-bit NF4 base + trainable LoRA
adapters, via TRL's SFTTrainer + PEFT.

Zero hardcoded hyperparameters — everything comes from the YAML config, so
"reproduce run N" literally means --config configs/exp_N.yaml. Data arrives only
through hash manifests (fetched + verified if the local copy is missing or
stale), prompts come from tinyllm.prompt (the ONE place they may live), and
every run logs params, git commit, loss curves and the adapter to MLflow
(MLFLOW_TRACKING_URI from the environment / .env — azureml:// since 2.8).

Era-2 additions — all config-driven, all defaulting to Era-1 behaviour so
exp_001–014 stay re-runnable:
  - model_revision       pin the exact HF commit
  - chat_format          True → {"prompt","completion"} rows built with the
                         tokenizer's chat template; TRL 1.12 then computes loss
                         on the completion only (verified: sft_trainer.py sets
                         completion_only_loss when both keys are present)
  - compute_dtype        auto | fp16 | fp32 — see resolve_compute_dtype()
  - gradient_checkpointing / max_length   ON + 640 for the 2B on a T4

CLI: uv run python -m tinyllm.train --config configs/exp_101.yaml
     [--max-steps N --limit N]                       smoke runs
     [--resume-from outputs/exp_101/run/checkpoint-500]
     [--log-merged]                                  winner only (~4–8 GB upload)
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
from tinyllm.prompt import chat_prompt, format_example


def set_seed(seed: int):  # same seed → same run → reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_compute_dtype(name: str) -> torch.dtype:
    """'auto' → bf16 on Ampere+ (compute capability 8.x), else fp32.

    NOT torch.cuda.is_bf16_supported(): since torch 2.3 it returns True on
    pre-Ampere GPUs because *emulated* bf16 counts (verified True on a GTX
    1650 Ti, sm_75 — the same generation as a T4). A T4 would then "train in
    bf16" through emulation at a crawl. fp16 / fp32 are explicit choices —
    §2.2: on a T4 run scripts/gen_smoke.py once, then set fp16 in the config."""
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_from_manifest(
    manifest_path: str,
    limit: int | None = None,
    eos: str = "",
    sender_id_frac: float = 0.0,
    seed: int = 42,
    tok=None,
    chat: bool = False,
):
    """Dataset by content hash, never by filename trust: verify the local file
    against the manifest and re-fetch from Blob on any mismatch.

    chat=True  → {"prompt","completion"} rows; TRL masks the prompt so the loss
                 lands on the JSON only. The prompt comes from chat_prompt(tok,…)
                 — the same helper model_eval --chat and serving use.
    chat=False → era-1 raw "text" rows (exp_009 remains re-runnable).

    `eos` is appended to every completion — without it the model never sees a
    stop signal (the first Era-1 sweep's outputs trailed junk). TRL would add
    it anyway when missing; explicit is clearer.
    `sender_id_frac` applies the DLT sender-ID augmentation to a seeded fraction
    of rows at load time (train split only; corpus-sourced UK rows are exempt —
    an Indian DLT prefix on a UK smishing text is an incoherent hybrid)."""
    from datasets import Dataset

    from tinyllm.augment import add_sender_id

    if chat and tok is None:
        raise ValueError("chat=True needs the tokenizer (tok=...)")
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
    out = []
    for r in rows:
        augmentable = not str(r.get("source", "")).startswith("mendeley")
        if sender_id_frac and augmentable and rng.random() < sender_id_frac:
            r = {**r, "sms": add_sender_id(r["sms"], rng)}
        if chat:
            out.append(
                {
                    "prompt": chat_prompt(tok, r["sms"]),
                    "completion": json.dumps(r["label"], ensure_ascii=False) + eos,
                }
            )
        else:
            out.append({"text": format_example(r)["text"] + eos})
    return Dataset.from_list(out)


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
) -> str:
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
    compute_dtype = resolve_compute_dtype(cfg.compute_dtype)

    mlflow.set_experiment("tinyllm-finetune")  # honours MLFLOW_TRACKING_URI
    with mlflow.start_run(run_name=os.path.basename(config_path)) as run:
        mlflow.log_params(flat_params(cfg))  # includes compute_dtype AS CONFIGURED
        mlflow.log_param("config_path", config_path)
        mlflow.log_param(
            "git_commit",
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        )
        mlflow.log_param(
            "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        # distinct key: MLflow refuses to overwrite a param ("compute_dtype" is
        # already logged from the config above — same key would crash the run)
        mlflow.log_param("compute_dtype_resolved", str(compute_dtype))

        bnb = BitsAndBytesConfig(  # the "Q" in QLoRA
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, quantization_config=bnb
        )  # qwen3_5 checkpoints auto-resolve to the TEXT-ONLY Qwen3_5ForCausalLM

        peft_cfg = LoraConfig(  # the "LoRA"
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )

        frac = cfg.augment.sender_id_frac if cfg.augment else 0.0
        ds_kw = dict(eos=tok.eos_token, tok=tok, chat=cfg.chat_format)
        val = (
            load_from_manifest(cfg.val_manifest, limit, **ds_kw)
            if cfg.val_manifest
            else None
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=load_from_manifest(
                cfg.dataset_manifest,
                limit,
                sender_id_frac=frac,
                seed=cfg.seed,
                **ds_kw,
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
                gradient_checkpointing=cfg.gradient_checkpointing,
                max_length=cfg.max_length,
                eval_strategy="epoch" if val is not None else "no",
                bf16=compute_dtype is torch.bfloat16,
                fp16=compute_dtype is torch.float16,
                seed=cfg.seed,
                logging_steps=10,
                report_to="mlflow",
            ),
        )
        if compute_dtype is torch.float16:
            # TRL 1.12 casts QLoRA adapter weights to bf16 (QLoRA-paper default,
            # sft_trainer.py "_is_quantized_model"). Under fp16 AMP the GradScaler
            # then dies at step 1 on pre-Ampere GPUs: "_amp_foreach_non_finite_
            # check_and_unscale_cuda not implemented for BFloat16" (reproduced on a
            # GTX 1650 Ti, sm_75 = T4). fp32 master weights + fp16 autocast is the
            # standard T4 recipe; the optimizer is created later, so this is safe.
            for p in trainer.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        if compute_dtype is torch.float16:
            # TRL 1.12 casts every trainable (LoRA) weight to bf16 on a quantized
            # model (QLoRA-paper default). Under fp16 AMP the GradScaler then
            # dies at step 1: "_amp_foreach_non_finite_check_and_unscale_cuda
            # not implemented for 'BFloat16'" (reproduced on a GTX 1650 Ti —
            # a T4 fails the same way). fp32 master weights + fp16 compute is
            # the standard AMP recipe; the adapters are tiny, so no VRAM cost.
            for param in trainer.model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
        if resume_from:  # restores weights + optimizer + scheduler + RNG state
            mlflow.log_param("resumed_from_checkpoint", resume_from)
        trainer.train(resume_from_checkpoint=resume_from)
        adapter_dir = os.path.join(out_root, "adapter")
        trainer.save_model(adapter_dir)
        mlflow.log_artifacts(adapter_dir, artifact_path="adapter")  # small, always

        # Merge adapters back into the base → one normal model (GGUF needs this).
        # Re-merge against a CLEAN fp32 base on CPU: merging into the 4-bit
        # quantized weights would bake quantization error into the artifact.
        # Free the trainer first — a 2B fp32 base is ~8 GB of host RAM and a
        # standard Colab VM has 12.7 GB.
        del trainer, model
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, dtype=torch.float32
        )
        merged_dir = os.path.join(out_root, "merged")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.to(torch.bfloat16 if compute_dtype is torch.bfloat16 else torch.float32)
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)  # GGUF conversion needs the tokenizer
        if log_merged:  # 2B fp32 ≈ 8 GB — sweep runs skip this; the winner logs it
            mlflow.log_artifacts(merged_dir, artifact_path="model")
        print(
            f"merged model + tokenizer → {merged_dir}"
            + (" (logged to MLflow)" if log_merged else "")
        )
        return run.info.run_id  # what the pipeline registers (Stage 3)


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
        help="trainer checkpoint dir, e.g. outputs/exp_101/run/checkpoint-500",
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
