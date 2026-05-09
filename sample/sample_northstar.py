"""Sample N completions per item from Northstar-CUA-Fast using vLLM.

Northstar-CUA-Fast is a Qwen3-VL-4B model fine-tuned with GRPO for GUI agent
tasks. It expects:
  * The Qwen3-VL chat template (auto-applied by vLLM via tokenizer_config.json).
  * A ``computer_use`` tool definition in the request, which renders into a
    ``<tools>...</tools>`` system block. The model replies with a
    ``<tool_call>{"name": "computer_use", "arguments": {...}}</tool_call>``
    block whose arguments include ``type`` (e.g. "click") and 0-999 normalized
    ``x`` / ``y`` integers (denormalize to pixels in ``filter_by_bbox.py``).
  * Image is placed before text in the user message (matches the model card
    snippet and the lightcone ``coordinate_scaling.py`` example).

References:
  * https://huggingface.co/Tzafon/Northstar-CUA-Fast (model card + chat template)
  * https://github.com/tzafon/lightcone/blob/main/examples/coordinate_scaling.py
  * https://github.com/tzafon/lightcone/blob/main/examples/_cua.py
    (TOOL = {type: "computer_use", display_width, display_height, environment};
     COORD_KEYS scales x/x1/x2 by display_width/1000.)

Each item is sampled n times in a single vLLM request (so vLLM expands n
internally and shares the prefill — much faster than n separate requests).

Usage:
    python sample/sample_northstar.py \\
        --input data/osatlas_hard_5k.jsonl \\
        --output sample/raw_samples.jsonl \\
        --n 8 --temperature 0.7 --top-p 0.9 \\
        --max-tokens 512 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# Tool spec lives in the repo-root single-source-of-truth module so the eval
# pipeline (eval/eval_screenspot_pro.py) renders the IDENTICAL schema.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from northstar_format import build_computer_use_tool  # noqa: E402

# vLLM / transformers are heavy and only needed at runtime on the H100.
# Import lazily so that --help and unit-style sanity checks work without GPU.

MODEL_ID = "Tzafon/Northstar-CUA-Fast"


def build_messages(instruction: str, image_path: str) -> list[dict[str, Any]]:
    """Build the OpenAI-style messages list. Image first, then text."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def gpu_util_string() -> str:
    """Cheap nvidia-smi snapshot. Returns 'n/a' on hosts without nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        # First GPU only
        first = out.splitlines()[0]
        util, mem_used, mem_total = [s.strip() for s in first.split(",")]
        return f"GPU {util}% util, {mem_used}/{mem_total} MiB"
    except Exception:
        return "GPU n/a"


def already_done_ids(output_path: Path) -> set[str]:
    """Resume support: skip items already present in output."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                done.add(str(obj["item_id"]))
            except Exception:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL with image_path, instruction, bbox, image_width, image_height")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL of {item_id, instruction, image_path, bbox, image_width, image_height, samples}")
    parser.add_argument("--n", type=int, default=8, help="Samples per item (default 8)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Items per vLLM .chat() call (each expands to n samples internally)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max sequence length. Increase if image tokens overflow.")
    parser.add_argument("--max-image-tokens", type=int, default=2048,
                        help="Cap visual tokens per image via mm_processor_kwargs.max_pixels heuristic.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only process the first N items (debug).")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import — keeps --help fast and lets the script exist on a non-GPU box.
    print("[setup] importing vllm...", flush=True)
    from vllm import LLM, SamplingParams  # type: ignore

    print(f"[setup] loading model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        # Cap per-image visual tokens to keep prefill cheap. Qwen3-VL uses
        # patch=16 with merge=2, so each (32-px)^2 image tile = 1 vision token.
        # max_pixels ~ max_image_tokens * 32 * 32.
        mm_processor_kwargs={
            "max_pixels": args.max_image_tokens * 32 * 32,
        },
        limit_mm_per_prompt={"image": 1},
    )

    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    done = already_done_ids(args.output)
    if done:
        print(f"[resume] skipping {len(done)} item_ids already in output", flush=True)

    total = count_lines(args.input)
    if args.limit is not None:
        total = min(total, args.limit)
    print(f"[setup] {total} items to process (n={args.n} -> {total * args.n} completions)", flush=True)

    out_fh = args.output.open("a", encoding="utf-8")
    processed = 0
    skipped = 0
    t0 = time.time()
    batch: list[dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal processed
        if not batch:
            return
        # Build per-item conversations + tool list. vLLM .chat() applies the
        # tokenizer's chat template (which is the Qwen3-VL template confirmed
        # in tokenizer_config.json on HF) and handles image loading.
        conversations = []
        tool_lists = []
        for item in batch:
            conversations.append(build_messages(item["instruction"], item["image_path"]))
            tool_lists.append([build_computer_use_tool(
                int(item["image_width"]), int(item["image_height"]),
            )])

        # vLLM accepts per-request tool lists via chat_template_kwargs.
        request_outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            tools=tool_lists,  # one tool list per conversation
            use_tqdm=False,
        )

        for item, ro in zip(batch, request_outputs):
            samples = [out.text for out in ro.outputs]
            rec = {
                "item_id": item["item_id"],
                "instruction": item["instruction"],
                "image_path": item["image_path"],
                "bbox": item["bbox"],
                "image_width": item["image_width"],
                "image_height": item["image_height"],
                "samples": samples,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1
        out_fh.flush()
        os.fsync(out_fh.fileno())
        batch.clear()

    try:
        for raw_item in iter_jsonl(args.input):
            if args.limit is not None and (processed + skipped + len(batch)) >= args.limit:
                break
            # Synthesize stable item_id if missing (use index hash of image_path+instruction).
            item_id = raw_item.get("item_id")
            if item_id is None:
                item_id = f"{raw_item['image_path']}::{hash(raw_item['instruction']) & 0xffffffff:08x}"
                raw_item["item_id"] = item_id
            if str(item_id) in done:
                skipped += 1
                continue
            batch.append(raw_item)
            if len(batch) >= args.batch_size:
                flush_batch()
                if processed % 50 == 0 or processed == total:
                    elapsed = time.time() - t0
                    rate = processed / max(elapsed, 1e-9)
                    eta = (total - processed) / max(rate, 1e-9)
                    print(
                        f"[progress] {processed}/{total} items "
                        f"({rate:.2f} items/s, ETA {eta/60:.1f} min) | "
                        f"{gpu_util_string()}",
                        flush=True,
                    )
        flush_batch()
    finally:
        out_fh.close()

    elapsed = time.time() - t0
    print(
        f"[done] {processed} items, skipped {skipped} (resume) "
        f"in {elapsed/60:.1f} min ({processed/max(elapsed,1e-9):.2f} items/s). "
        f"Output: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
