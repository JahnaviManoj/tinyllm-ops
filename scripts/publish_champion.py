"""4.6, once by hand for the current champion (later promotions do this inside
promote.py): attach the local GGUFs to the champion's run if missing, copy the
served quant to the fixed Blob address, and mint the read-only URLs the
container boots from (1-year SAS, written to .env — gitignored).

  uv run python scripts/publish_champion.py            # does it
  uv run python scripts/publish_champion.py --dry-run  # shows what it would do
"""

import argparse
import datetime as dt
import os
import re
import sys

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinyllm.manifest import CONTAINER
from tinyllm.quantize import quantize_to_gguf
from tinyllm.registry import (
    CHAMPION,
    CHAMPION_BLOB,
    CHAMPION_INFO_BLOB,
    MODEL_NAME,
    SERVED_QUANT,
    get_version_by_alias,
    publish_champion_gguf,
)

MERGED = "outputs/{exp}/merged"


def sas_url(blob_path: str, days: int = 365) -> str:
    from azure.storage.blob import BlobSasPermissions, generate_blob_sas

    cs = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    account = re.search(r"AccountName=([^;]+)", cs).group(1)
    key = re.search(r"AccountKey=([^;]+)", cs).group(1)
    token = generate_blob_sas(
        account_name=account,
        container_name=CONTAINER,
        blob_name=blob_path,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        expiry=dt.datetime.now(dt.UTC) + dt.timedelta(days=days),
    )
    return f"https://{account}.blob.core.windows.net/{CONTAINER}/{blob_path}?{token}"


def write_env(values: dict[str, str], path: str = ".env") -> None:
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
    lines = [ln for ln in lines if ln.split("=", 1)[0] not in values]
    lines += [f'{k}="{v}"' for k, v in values.items()]  # quoted: SAS URLs contain "&"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--exp", default="exp_111", help="local merged dir for the GGUF attach"
    )
    ap.add_argument("--quant", default=SERVED_QUANT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    client = MlflowClient()

    mv = get_version_by_alias(client, MODEL_NAME, CHAMPION)
    have = {a.path for a in client.list_artifacts(mv.run_id)}
    print(f"@champion = v{mv.version} (run {mv.run_id[:8]}); run has {sorted(have)}")
    if args.dry_run:
        print(
            f"would: attach gguf/ if missing → copy gguf/{args.quant}.gguf to "
            f"{CONTAINER}/{CHAMPION_BLOB} (+ {CHAMPION_INFO_BLOB}) → write MODEL_URL etc. to .env"
        )
        return

    if "gguf" not in have:
        print(f"attaching local GGUFs to run {mv.run_id[:8]} ...", flush=True)
        quantize_to_gguf(MERGED.format(exp=args.exp), run_id=mv.run_id)

    info = publish_champion_gguf(mv, args.quant, client=client)
    print(
        f"published {CONTAINER}/{CHAMPION_BLOB}: v{info['model_version']} {info['quant']} "
        f"{info['bytes'] / 1e9:.2f} GB sha {info['sha256'][:12]}…"
    )

    values = {
        "MODEL_URL": sas_url(CHAMPION_BLOB),
        "MODEL_INFO_URL": sas_url(CHAMPION_INFO_BLOB),
        "MODEL_SHA256": info["sha256"],
    }
    write_env(values)
    print(
        "wrote MODEL_URL / MODEL_INFO_URL / MODEL_SHA256 to .env (read-only SAS, 1 year; "
        "signatures not shown)"
    )


if __name__ == "__main__":
    main()
