"""Snapshot freezer for the streaming RSSD pipeline.

The sampler daemon writes ``samples/shard_<id>/sft.jsonl`` continuously. To
launch a training run reproducibly, we freeze the current state of all shards
into ``snapshots/run_<id>/``. The frozen snapshot is the input to
``train_lora.py`` — sampling continues writing to the source dir uninterrupted.

Snapshot layout:
    snapshots/run_<id>/
      sft.jsonl              # concatenation of every shard's sft.jsonl
      snapshot_meta.json     # { run_id, created_at, source_shards, total_sft_items }

We snapshot by READING the source files (not ``cp``) — this way we can both
(a) skip blank/garbage lines and (b) emit a single concatenated file the
trainer ingests directly, without a glob. The source dir is untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _shard_dirs(samples_dir: Path) -> list[Path]:
    return sorted(
        p for p in samples_dir.iterdir()
        if p.is_dir() and p.name.startswith("shard_")
    )


def freeze_snapshot(
    samples_dir: Path,
    snapshots_dir: Path,
    run_id: str,
) -> Path:
    """Snapshot every shard's ``sft.jsonl`` into ``snapshots/run_<id>/``.

    Args:
        samples_dir: source directory; must contain ``shard_*/`` subdirs.
        snapshots_dir: parent directory under which to create ``run_<id>/``.
            Will be created (incl. parents) if missing.
        run_id: short identifier (typically ``HHMM``); becomes the dir suffix.

    Returns:
        Path to the created ``snapshots/run_<id>/`` dir.

    Raises:
        FileNotFoundError: if ``samples_dir`` has no ``shard_*`` subdirs.
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples dir not found: {samples_dir}")

    shards = _shard_dirs(samples_dir)
    if not shards:
        raise FileNotFoundError(
            f"no shard_* subdirs in {samples_dir} (sampler hasn't written anything yet)"
        )

    snap = Path(snapshots_dir) / f"run_{run_id}"
    snap.mkdir(parents=True, exist_ok=True)

    out_jsonl = snap / "sft.jsonl"
    total = 0
    source_shards: list[str] = []
    with out_jsonl.open("w", encoding="utf-8") as out_fh:
        for shard in shards:
            source_shards.append(shard.name)
            sft = shard / "sft.jsonl"
            if not sft.is_file():
                # Sampler may not have produced any rows yet; tolerate.
                continue
            with sft.open("r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    out_fh.write(line + "\n")
                    total += 1

    meta = {
        "run_id": run_id,
        "created_at": time.time(),
        "source_shards": source_shards,
        "total_sft_items": total,
        "samples_dir": str(samples_dir.resolve()),
    }
    (snap / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    return snap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Freeze a samples/ snapshot for training.")
    p.add_argument("--samples-dir", type=Path, default=Path("samples"))
    p.add_argument("--snapshots-dir", type=Path, default=Path("snapshots"))
    p.add_argument("--run-id", required=True,
                   help="Short tag, typically HHMM (e.g. '1130')")
    args = p.parse_args()

    snap = freeze_snapshot(args.samples_dir, args.snapshots_dir, args.run_id)
    meta = json.loads((snap / "snapshot_meta.json").read_text())
    print(f"Froze snapshot -> {snap}")
    print(f"  source_shards   : {meta['source_shards']}")
    print(f"  total_sft_items : {meta['total_sft_items']}")


if __name__ == "__main__":
    _main()
