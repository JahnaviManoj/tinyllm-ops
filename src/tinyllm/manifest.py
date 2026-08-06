"""Dataset versioning via hash manifests (tutorial 1.7).

Data never enters git. A tiny manifest (committed) records the SHA-256 of the
JSONL plus the config/seed that produced it; the bytes live in Azure Blob at a
path named by their own hash. Anyone — including CI — can re-download the
exact bytes and verify them. This is DVC's mechanism, hand-rolled (~50 lines)
so there's one less tool and full understanding of the design; see
docs/decisions.md for the alternatives (DVC, lakeFS, HF Hub) and why not.

  publish:  uv run python -m tinyllm.manifest publish \
                --data data/generated/train.jsonl --name train --version v1 \
                --config configs/datagen_v1.yaml --seed 42
  fetch:    uv run python -m tinyllm.manifest fetch \
                --manifest manifests/train_v1.json --out data/generated/train.jsonl

publish needs AZURE_STORAGE_CONNECTION_STRING in .env (--dry-run skips the
upload and just writes the manifest). fetch verifies the SHA-256 and refuses
corrupted or wrong-version bytes.
"""

import argparse
import hashlib
import json
import os

CONTAINER = "artifacts"


def _conn_str() -> str:
    cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not cs:
        raise SystemExit(
            "AZURE_STORAGE_CONNECTION_STRING is not set — add it to .env "
            "(az storage account show-connection-string ...)"
        )
    return cs


def publish(
    data_path: str,
    name: str,
    version: str,
    config: str | None,
    seed: int | None,
    dry_run: bool = False,
) -> str:
    data = open(data_path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    with open(data_path) as f:
        num = sum(1 for _ in f)
    manifest = {
        "name": name,
        "version": version,
        "sha256": sha,
        "num_examples": num,
        "blob_path": f"datasets/{sha}/{os.path.basename(data_path)}",
        "generator_config": config,
        "seed": seed,
    }
    if not dry_run:
        from azure.storage.blob import BlobServiceClient

        svc = BlobServiceClient.from_connection_string(_conn_str())
        try:
            svc.create_container(CONTAINER)
        except Exception:
            pass  # already exists
        blob = svc.get_blob_client(CONTAINER, manifest["blob_path"])
        blob.upload_blob(
            data, overwrite=True
        )  # content-addressed: same bytes → same path
    os.makedirs("manifests", exist_ok=True)
    out = f"manifests/{name}_{version}.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        f"{'DRY RUN — not uploaded' if dry_run else 'uploaded'} "
        f"{num} examples, sha256 {sha[:12]}… → {out}"
    )
    return out


def fetch_dataset(manifest_path: str, out_path: str):
    from azure.storage.blob import BlobClient

    manifest = json.load(open(manifest_path))
    blob = BlobClient.from_connection_string(
        _conn_str(), CONTAINER, manifest["blob_path"]
    )
    data = blob.download_blob().readall()
    actual = hashlib.sha256(data).hexdigest()
    if actual != manifest["sha256"]:
        raise SystemExit(
            f"HASH MISMATCH for {manifest_path}: data corrupted or wrong version "
            f"(expected {manifest['sha256'][:12]}…, got {actual[:12]}…)"
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "wb").write(data)
    print(
        f"fetched + verified {manifest['num_examples']} examples "
        f"({manifest['name']} {manifest['version']}) → {out_path}"
    )


def main():
    from dotenv import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(
        description="Publish or fetch hash-manifested datasets."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--data", required=True)
    pub.add_argument("--name", required=True)
    pub.add_argument("--version", required=True)
    pub.add_argument("--config", default=None)
    pub.add_argument("--seed", type=int, default=None)
    pub.add_argument("--dry-run", action="store_true")
    fet = sub.add_parser("fetch")
    fet.add_argument("--manifest", required=True)
    fet.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "publish":
        publish(
            args.data, args.name, args.version, args.config, args.seed, args.dry_run
        )
    else:
        fetch_dataset(args.manifest, args.out)


if __name__ == "__main__":
    main()
