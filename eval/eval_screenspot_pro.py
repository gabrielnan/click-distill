"""Evaluate Northstar-CUA-Fast (or a LoRA-tuned variant) on ScreenSpot-Pro.

Click accuracy = predicted (x, y) lies inside the ground-truth bbox.

Usage:
    # 10-item sanity check (do this FIRST on a new GPU box)
    python eval/eval_screenspot_pro.py \
        --model Tzafon/Northstar-CUA-Fast --limit 10

    # Full run (1.5k items, ~10 min on 1xH100 with vLLM batching)
    python eval/eval_screenspot_pro.py \
        --model Tzafon/Northstar-CUA-Fast \
        --output eval/results/baseline.json

    # With a tuned LoRA adapter
    python eval/eval_screenspot_pro.py \
        --model Tzafon/Northstar-CUA-Fast \
        --lora ./outputs/rssd-lora \
        --output eval/results/tuned.json

Backends:
    --backend vllm         (default) batched inference, fast
    --backend transformers fallback if vLLM unavailable; one-at-a-time

Image preprocessing follows the reference Qwen3-VL ScreenSpot-Pro setup
(`smart_resize` factor=32, max_pixels controls memory). Northstar inherits
the Qwen3-VL processor.

Coordinate space: model emits 0-1000 normalised; we convert to source-image
pixels and check inclusion in the ground-truth xyxy pixel bbox.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Use the repo-root single-source-of-truth for tool spec + parser. This is
# the same module sample/sample_northstar.py imports — eval and sampling MUST
# render identical tool schemas, otherwise eval will under-count valid clicks.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from northstar_format import (  # noqa: E402
    build_computer_use_tool,
    parse_action,
    denormalize,
)

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "screenspot_pro"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_items(data_dir: Path, limit: int | None = None) -> list[dict]:
    jsonl = data_dir / "screenspot_pro.jsonl"
    if not jsonl.is_file():
        sys.exit(
            f"Missing {jsonl}.\n"
            f"Run: python eval/download_screenspot_pro.py --out {data_dir}"
        )
    items: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class VLLMBackend:
    """Batched vLLM backend. Greedy by default."""

    def __init__(self, model_path: str, lora_path: str | None, max_pixels: int,
                 max_new_tokens: int, max_num_seqs: int):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        self.lora_request = None

        kwargs = dict(
            model=model_path,
            gpu_memory_utilization=0.92,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"min_pixels": 32 * 32, "max_pixels": max_pixels},
            trust_remote_code=True,
            dtype="bfloat16",
        )
        if lora_path:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = 64  # PLAN.md uses r=32; 64 leaves headroom
            self.lora_request = LoRARequest("rssd-lora", 1, lora_path)

        self.llm = LLM(**kwargs)
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0,
                                       max_tokens=max_new_tokens)
        self.max_pixels = max_pixels

    def generate(self, prompts: list[dict], tools: list[list[dict]] | None = None) -> list[str]:
        kwargs = {"sampling_params": self.sampling}
        if self.lora_request is not None:
            kwargs["lora_request"] = self.lora_request
        if tools is not None:
            kwargs["tools"] = tools
        outs = self.llm.chat(prompts, **kwargs)
        return [o.outputs[0].text for o in outs]


class TransformersBackend:
    """Fallback path. One image at a time. Slow but unblocks if vLLM breaks."""

    def __init__(self, model_path: str, lora_path: str | None, max_pixels: int,
                 max_new_tokens: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, max_pixels=max_pixels,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[dict], tools: list[list[dict]] | None = None) -> list[str]:
        out_texts: list[str] = []
        for idx, messages in enumerate(prompts):
            tool_arg = tools[idx] if tools is not None else None
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=tool_arg,
            )
            # Pull the PIL image out of the message struct.
            image = None
            for m in messages:
                if m["role"] != "user":
                    continue
                for c in m["content"]:
                    if c.get("type") == "image":
                        image = c["image"]
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt",
            ).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,  # ignored when do_sample=False
                )
            trimmed = gen[:, inputs["input_ids"].shape[1]:]
            out_texts.append(self.processor.batch_decode(
                trimmed, skip_special_tokens=False,
            )[0])
        return out_texts


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_messages(image, instruction: str) -> list[dict]:
    """Chat-completions-style messages with an inline image. vLLM and
    transformers both accept this shape via apply_chat_template / chat().

    No explicit system prompt: the chat template renders the per-request tool
    list (passed via ``tools=``) into a synthesized system block. This matches
    sample/sample_northstar.py exactly so that eval and sampling see the same
    rendered prompt.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_prompt(image, instruction: str, image_width: int, image_height: int) -> tuple[list[dict], list[dict]]:
    """Convenience helper: returns (messages, tools) for one item, mirroring
    sample/sample_northstar.py's per-item shape. Useful for sanity checks."""
    return (
        build_messages(image, instruction),
        [build_computer_use_tool(image_width, image_height)],
    )


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
def evaluate(items: list[dict], backend, data_dir: Path, batch_size: int,
             progress: bool = True) -> list[dict]:
    from PIL import Image
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    results: list[dict] = []

    def chunked(xs: list, n: int) -> Iterable[list]:
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    iterator = chunked(items, batch_size)
    if progress:
        iterator = tqdm(list(iterator), desc="eval")

    for batch in iterator:
        messages_batch = []
        tools_batch = []
        for item in batch:
            img_path = data_dir / item["image"]
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = item["img_size"]
            messages_batch.append(build_messages(image, item["instruction"]))
            tools_batch.append([build_computer_use_tool(int(img_w), int(img_h))])

        raws = backend.generate(messages_batch, tools=tools_batch)
        for item, raw in zip(batch, raws):
            parsed = parse_action(raw)
            img_w, img_h = item["img_size"]
            if parsed is None:
                results.append({
                    **item,
                    "raw_response": raw,
                    "pred_norm": None,
                    "pred_pixel": None,
                    "correct": False,
                    "wrong_format": True,
                })
                continue
            x_norm, y_norm = parsed["x"], parsed["y"]
            px, py = denormalize(x_norm, y_norm, int(img_w), int(img_h))
            x1, y1, x2, y2 = item["bbox"]
            correct = (x1 <= px <= x2) and (y1 <= py <= y2)
            results.append({
                **item,
                "raw_response": raw,
                "pred_norm": [x_norm, y_norm],
                "pred_pixel": [px, py],
                "correct": correct,
                "wrong_format": False,
            })
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_format = sum(1 for r in results if r["wrong_format"])

    by_app: dict[str, list[dict]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_ui: dict[str, list[dict]] = defaultdict(list)
    by_platform: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_app[r.get("application") or "unknown"].append(r)
        by_group[r.get("group") or "unknown"].append(r)
        by_ui[r.get("ui_type") or "unknown"].append(r)
        by_platform[r.get("platform") or "unknown"].append(r)

    def acc(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "acc": (sum(1 for r in rs if r["correct"]) / len(rs)) if rs else 0.0,
            "wrong_format": sum(1 for r in rs if r["wrong_format"]),
        }

    return {
        "overall": {
            "n": n,
            "acc": (n_correct / n) if n else 0.0,
            "wrong_format": n_format,
        },
        "by_application": {k: acc(v) for k, v in sorted(by_app.items())},
        "by_group": {k: acc(v) for k, v in sorted(by_group.items())},
        "by_ui_type": {k: acc(v) for k, v in sorted(by_ui.items())},
        "by_platform": {k: acc(v) for k, v in sorted(by_platform.items())},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF id or local path (e.g. Tzafon/Northstar-CUA-Fast)")
    p.add_argument("--lora", default=None,
                   help="Optional LoRA adapter path to overlay on the base model")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"ScreenSpot-Pro data dir (default: {DEFAULT_DATA})")
    p.add_argument("--output", type=Path, default=Path("eval/results/results.json"),
                   help="Output JSON path (per-item preds + aggregated metrics)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N items (for debugging / sanity check)")
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for vLLM. Ignored by transformers backend.")
    p.add_argument("--max-pixels", type=int, default=12_845_056,
                   help="Qwen3-VL smart_resize cap. Default matches reference "
                        "Qwen3-VL ScreenSpot-Pro eval (~12.8M pixels).")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Tool-call output is short; 128 is plenty.")
    p.add_argument("--max-num-seqs", type=int, default=16,
                   help="vLLM max concurrent sequences (memory cap).")
    args = p.parse_args()

    print(f"Loading items from {args.data}")
    items = load_items(args.data, limit=args.limit)
    print(f"  {len(items)} items "
          f"({'sanity check' if args.limit else 'full eval'})")

    print(f"Loading {args.backend} backend: model={args.model} lora={args.lora}")
    if args.backend == "vllm":
        backend = VLLMBackend(args.model, args.lora, args.max_pixels,
                              args.max_new_tokens, args.max_num_seqs)
        batch_size = args.batch_size
    else:
        backend = TransformersBackend(args.model, args.lora, args.max_pixels,
                                      args.max_new_tokens)
        batch_size = 1

    t0 = time.time()
    results = evaluate(items, backend, args.data, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"\nEval done in {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/item)")

    metrics = aggregate(results)
    print(f"\n=== Overall: {metrics['overall']['acc']:.4f} "
          f"({metrics['overall']['n']} items, "
          f"{metrics['overall']['wrong_format']} wrong-format) ===")
    print("\nBy group:")
    for k, v in metrics["by_group"].items():
        print(f"  {k:12s} acc={v['acc']:.4f}  n={v['n']:4d}  wf={v['wrong_format']}")
    print("\nBy application (top 5 / bottom 5):")
    apps = sorted(metrics["by_application"].items(), key=lambda kv: kv[1]["acc"])
    for k, v in apps[-5:][::-1]:
        print(f"  {k:20s} acc={v['acc']:.4f}  n={v['n']}")
    print("  ...")
    for k, v in apps[:5]:
        print(f"  {k:20s} acc={v['acc']:.4f}  n={v['n']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "lora": args.lora,
        "n_items": len(results),
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "predictions": results,
    }
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
