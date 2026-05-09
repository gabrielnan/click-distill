"""Streaming RSSD sampler for Northstar-CUA-Fast (Qwen3-VL-4B + GRPO).

Two operating modes:

* **Legacy raw mode** (``--output raw_samples.jsonl``) — kept for back-compat,
  writes one row per item with all 8 raw completions. Use ``filter_by_bbox.py``
  downstream to produce SFT data.

* **Streaming shard mode** (``--shard-dir samples/shard_<id>``, default if
  ``--num-shards`` is given) — daemon-style. Filters input by hash-mod
  (``line_index % num_shards == shard_id``), parses + bbox-checks each item's
  completions immediately, writes only the winning completion to a per-shard
  ``sft.jsonl``. Per-item synchronous flush (write-then-fsync) means a reader
  always sees complete records. See ``sample/README.md`` for the on-disk
  layout and the ``sample/launch_shards.sh`` orchestrator.

What stays the same in both modes:
  * Qwen3-VL chat template (auto-applied by vLLM via tokenizer_config.json).
  * A ``computer_use`` tool definition rendered into the system block.
  * Image-before-text in the user message.
  * Per-item ``n`` completions sampled via vLLM's internal n expansion (shared
    prefill — much cheaper than n separate requests).

References:
  * https://huggingface.co/Tzafon/Northstar-CUA-Fast (model card + chat template)
  * https://github.com/tzafon/lightcone/blob/main/examples/coordinate_scaling.py
  * https://github.com/tzafon/lightcone/blob/main/examples/_cua.py
    (TOOL = {type: "computer_use", display_width, display_height, environment};
     COORD_KEYS scales x/x1/x2 by display_width/1000.)

Usage (streaming shard mode):
    python sample/sample_northstar.py \\
        --input data/osatlas_hard_50k.jsonl \\
        --shard-dir samples/shard_0 \\
        --shard 0 --num-shards 4 \\
        --n 8 --temperature 0.7 --top-p 0.9 --max-tokens 512 --batch-size 64

Usage (legacy raw mode, single GPU):
    python sample/sample_northstar.py \\
        --input data/osatlas_hard_5k.jsonl \\
        --output sample/raw_samples.jsonl \\
        --n 8 --temperature 0.7 --batch-size 64
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


def build_messages(instruction: str, image_path: str, tool_spec_dict=None) -> list[dict[str, Any]]:
    """Build the OpenAI-style messages list. Image first, then text.
    If tool_spec_dict is given, prepend it as a system message text (bypasses HF validator)."""
    import json as _json
    msgs = []
    if tool_spec_dict is not None:
        msgs.append({
            "role": "system",
            "content": (
                "You are a computer-use agent. To take an action, respond with: "
                "<tool_call>{\"name\": \"computer_use\", \"arguments\": {\"type\": \"click\", \"x\": <int 0-999>, \"y\": <int 0-999>}}</tool_call>. "
                "Tool spec: " + _json.dumps(tool_spec_dict)
            ),
        })
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file://" + image_path}},
            {"type": "text", "text": instruction},
        ],
    })
    return msgs


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file. Tags each yielded dict with ``_line_index`` so
    downstream filters (sharding, dedupe) can address the source row by its
    1:1 line position. We use the line index — not a hash of fields — as the
    canonical work-distribution key so adding/removing fields never reshuffles
    which shard owns which item.
    """
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("_line_index", idx)
            yield obj


def iter_shard(
    items: Iterator[dict[str, Any]],
    *,
    shard: int,
    num_shards: int,
) -> Iterator[dict[str, Any]]:
    """Hash-mod work distribution: shard X owns items where ``index % N == X``.

    The "index" is the input position (0-based) — read from ``_line_index`` if
    present (set by :func:`iter_jsonl`), else assigned by enumeration order.

    This is intentionally trivial: no claims, no coordinator, no IPC. As long
    as every shard reads the SAME input file, two shards cannot pick the same
    item.

    Raises ``ValueError`` for invalid configs (negative shard, num_shards <= 0,
    shard >= num_shards). These are caller bugs, not data conditions.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if shard < 0 or shard >= num_shards:
        raise ValueError(
            f"shard must be in [0, {num_shards}), got {shard}"
        )
    for fallback_idx, item in enumerate(items):
        idx = item.get("_line_index", fallback_idx)
        if idx % num_shards == shard:
            yield item


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


# ---------------------------------------------------------------------------
# Streaming writer (Component 2: per-item synchronous flush)
# ---------------------------------------------------------------------------
# We import the canonical parser at the top of the writer module so test code
# can monkey-patch it if needed. The parser handles all 4 Northstar output
# formats — see ``northstar_format.parse_action`` for the trial order.
from northstar_format import parse_action, denormalize  # noqa: E402

# Action types we count as "click-ish" SFT signal. ``drag`` etc. are filtered
# out — we're distilling click-target accuracy, not multi-step interaction.
_POINT_ACTION_TYPES = {"click", "double_click", "triple_click", "right_click"}


def _bbox_hits(
    completion: str,
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
) -> bool:
    """Parse a completion + check whether its predicted click lands in the bbox.

    Mirrors ``filter_by_bbox.parse_click`` + ``in_bbox`` exactly so streaming
    mode matches the legacy two-stage pipeline byte-for-byte.
    """
    parsed = parse_action(completion)
    if parsed is None:
        return False
    if parsed["type"] not in _POINT_ACTION_TYPES:
        return False
    px, py = denormalize(int(parsed["x"]), int(parsed["y"]), image_width, image_height)
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (tmp + rename + fsync).

    Used for ``status.json`` snapshots — readers (the dashboard) can then
    ``json.load`` without ever seeing a half-written file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class StreamingWriter:
    """Owns one shard's on-disk state. Thread/process safety: this class is
    designed for SINGLE-WRITER use (one writer per shard dir). Multiple shards
    must point at distinct dirs.

    Per-item flow (``write_item``):
      1. Iterate completions, find the FIRST one whose click hits the bbox.
      2. If a hit was found: append to ``sft.jsonl``, fsync.
      3. If no hit: append item_id to ``zero_hits.txt``, fsync.
      4. Append item_id to ``done_ids.txt``, fsync.   ← AFTER step 2 always
      5. Re-write ``status.json`` (atomic).

    Crash-safety contract: ``done_ids.txt`` never references an item whose SFT
    row didn't make it to disk. Worst case a crash duplicates an SFT row on
    resume (idempotent for the trainer — ``item_id`` dedupes).
    """

    def __init__(
        self,
        shard_dir: Path,
        *,
        shard_id: int,
        gpu_id: str,
        n_per_item: int,
        temperature: float,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_id = shard_id
        self.gpu_id = gpu_id
        self.n_per_item = n_per_item
        self.temperature = temperature

        self.sft_path = self.shard_dir / "sft.jsonl"
        self.done_path = self.shard_dir / "done_ids.txt"
        self.zero_hits_path = self.shard_dir / "zero_hits.txt"
        self.status_path = self.shard_dir / "status.json"
        self.pid_path = self.shard_dir / "pid"

        # Counters survive across calls. Initialised by inspecting whatever's
        # already on disk so resume keeps the running totals correct.
        self.items_done = self._count_lines(self.done_path)
        self.zero_hit_items = self._count_lines(self.zero_hits_path)
        self.sft_items = self._count_lines(self.sft_path)

        self._started_at = time.time()
        self._last_write_at = self._started_at
        self._write_status_snapshot()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as fh:
            return sum(1 for _ in fh)

    @staticmethod
    def _append_line(path: Path, text: str) -> None:
        """Append a single line + fsync. Newline is added by the caller's text."""
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    def _append_done_id(self, item_id: str) -> None:
        self._append_line(self.done_path, f"{item_id}\n")

    def _append_zero_hit(self, item_id: str) -> None:
        self._append_line(self.zero_hits_path, f"{item_id}\n")

    def _append_sft(self, record: dict) -> None:
        self._append_line(self.sft_path, json.dumps(record, ensure_ascii=False) + "\n")

    def _write_status_snapshot(self) -> None:
        payload = {
            "shard_id": self.shard_id,
            "gpu_id": self.gpu_id,
            "pid": os.getpid(),
            "started_at": self._started_at,
            "last_write_at": self._last_write_at,
            "items_done": self.items_done,
            "sft_items": self.sft_items,
            "zero_hit_items": self.zero_hit_items,
            "n_per_item": self.n_per_item,
            "temperature": self.temperature,
        }
        _atomic_write_json(self.status_path, payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_skip_set(self) -> set[str]:
        """Return the set of item_ids the shard already finished — used by
        ``--resume`` to filter the input stream.

        Includes both items that produced an SFT row and items that hit 0/8
        (the latter are PERMANENTLY skipped — re-sampling won't help).
        """
        skip: set[str] = set()
        for path in (self.done_path, self.zero_hits_path):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        skip.add(line)
        return skip

    def write_item(self, item: dict, completions: list[str]) -> int:
        """Process one item's batch of completions. Returns the total hit count
        (NOT the number of SFT rows written — we always write at most one).

        The return value is for stats only; on-disk state captures the full
        decision (sft.jsonl / zero_hits.txt / done_ids.txt).
        """
        item_id = str(item["item_id"])
        bbox = item["bbox"]
        img_w = int(item["image_width"])
        img_h = int(item["image_height"])

        # Count all hits for stats but write only the FIRST.
        n_hits = 0
        first_hit_completion: str | None = None
        for completion in completions:
            if _bbox_hits(completion, bbox, img_w, img_h):
                n_hits += 1
                if first_hit_completion is None:
                    first_hit_completion = completion

        # Step 1+2: SFT row first (if any), THEN done_ids / zero_hits.
        if first_hit_completion is not None:
            record = {
                "item_id": item_id,
                "image_path": item["image_path"],
                "instruction": item["instruction"],
                "completion": first_hit_completion,
            }
            self._append_sft(record)
            self.sft_items += 1
        else:
            self._append_zero_hit(item_id)
            self.zero_hit_items += 1

        # Step 3: done_ids — guarantee comes from append-order: SFT row is
        # already on disk + fsynced before we record completion.
        self._append_done_id(item_id)
        self.items_done += 1

        # Step 4: bookkeeping.
        self._last_write_at = time.time()
        self._write_status_snapshot()
        return n_hits

    def write_pid(self) -> None:
        """Write our PID to ``shard_dir/pid`` so the orchestrator can kill us."""
        self.pid_path.write_text(f"{os.getpid()}\n")


def plan_shard_work(
    input_path: Path,
    *,
    shard: int,
    num_shards: int,
    skip_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """End-to-end input plan for one shard: stream → hash-mod → skip-resume.

    Composing :func:`iter_jsonl` + :func:`iter_shard` + a skip-set filter into
    one call so the main loop and tests share the exact same logic.
    """
    skip = skip_ids or set()
    sharded = iter_shard(iter_jsonl(input_path), shard=shard, num_shards=num_shards)
    for item in sharded:
        if str(item.get("item_id", "")) in skip:
            continue
        yield item


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
    # Output: legacy raw mode OR streaming shard mode (mutually exclusive).
    parser.add_argument("--output", type=Path, default=None,
                        help="(legacy raw mode) JSONL of {item_id, ..., samples}")
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help="(streaming mode) per-shard dir, e.g. samples/shard_0/")
    parser.add_argument("--shard", type=int, default=0,
                        help="This shard's id (0..num_shards-1).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards. shard X processes lines where idx %% N == X.")
    parser.add_argument("--gpu-id", type=str, default=None,
                        help="GPU id label for status.json (display only). Set CUDA_VISIBLE_DEVICES outside.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="(streaming mode) skip items already in done_ids/zero_hits (default).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="(streaming mode) ignore prior shard state.")
    parser.add_argument("--no-stop", action="store_true",
                        help="(streaming mode) loop forever — re-process input from the top.")
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

    if (args.output is None) == (args.shard_dir is None):
        sys.exit(
            "Pass exactly one of --output (legacy raw mode) or --shard-dir "
            "(streaming mode)."
        )
    if args.shard < 0 or args.shard >= args.num_shards:
        sys.exit(
            f"--shard {args.shard} out of range for --num-shards {args.num_shards}"
        )

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    if args.output is not None:
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
        allowed_local_media_path="/",
        enforce_eager=True,
    )

    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    if args.shard_dir is not None:
        _run_streaming_shard(args, llm, sampling)
    else:
        _run_legacy_raw(args, llm, sampling)


def _ensure_item_id(item: dict[str, Any]) -> str:
    """Return a stable item_id, synthesizing one if missing. Uses the
    ``_line_index`` attached by :func:`iter_jsonl` so item_ids are byte-stable
    across runs of the same input."""
    item_id = item.get("item_id")
    if item_id is None:
        if "_line_index" in item:
            item_id = f"line_{item['_line_index']}"
        else:
            item_id = f"{item['image_path']}::{hash(item['instruction']) & 0xffffffff:08x}"
        item["item_id"] = item_id
    return str(item_id)


def _run_legacy_raw(args, llm, sampling) -> None:
    """Original single-GPU mode: write all 8 raw completions per item to a
    single JSONL. Downstream filtering happens in ``filter_by_bbox.py``."""
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
        conversations = []
        tool_lists = []
        for item in batch:
            conversations.append(build_messages(item["instruction"], item["image_path"], tool_spec_dict=build_computer_use_tool(int(item["image_width"]), int(item["image_height"]))["function"]))
            tool_lists.append([build_computer_use_tool(
                int(item["image_width"]), int(item["image_height"]),
            )])
        request_outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            # tools=tool_lists,  # disabled: HF validator rejects custom tool spec
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
            item_id = _ensure_item_id(raw_item)
            if item_id in done:
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


def _run_streaming_shard(args, llm, sampling) -> None:
    """Streaming-shard mode: hash-mod input, eager filter, per-item flush.

    The loop runs until input is exhausted (one pass) unless ``--no-stop`` is
    set, in which case it loops forever — useful for the daemon launcher when
    we want a shard to keep re-sampling already-zero-hit items at higher temp
    in a later iteration. (Default: single pass.)
    """
    gpu_id = args.gpu_id
    if gpu_id is None:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.shard))
    writer = StreamingWriter(
        args.shard_dir,
        shard_id=args.shard,
        gpu_id=gpu_id,
        n_per_item=args.n,
        temperature=args.temperature,
    )
    writer.write_pid()

    skip = writer.load_skip_set() if args.resume else set()
    if skip:
        print(f"[resume] shard {args.shard}: skipping {len(skip)} prior item_ids", flush=True)

    t0 = time.time()
    batch: list[dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch:
            return
        conversations = []
        tool_lists = []
        for item in batch:
            conversations.append(build_messages(item["instruction"], item["image_path"], tool_spec_dict=build_computer_use_tool(int(item["image_width"]), int(item["image_height"]))["function"]))
            tool_lists.append([build_computer_use_tool(
                int(item["image_width"]), int(item["image_height"]),
            )])
        request_outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            # tools=tool_lists,  # disabled: HF validator rejects custom tool spec
            use_tqdm=False,
        )
        for item, ro in zip(batch, request_outputs):
            completions = [out.text for out in ro.outputs]
            writer.write_item(item, completions)
        batch.clear()

    pass_idx = 0
    while True:
        pass_idx += 1
        n_seen_this_pass = 0
        for item in plan_shard_work(
            args.input, shard=args.shard, num_shards=args.num_shards,
            skip_ids=skip,
        ):
            _ensure_item_id(item)
            if args.limit is not None and writer.items_done >= args.limit:
                break
            batch.append(item)
            n_seen_this_pass += 1
            if len(batch) >= args.batch_size:
                flush_batch()
                # Refresh skip set so within a single launch we never re-process
                # something we just wrote.
                skip = writer.load_skip_set() if args.resume else skip
                elapsed = time.time() - t0
                rate = writer.items_done / max(elapsed, 1e-9)
                print(
                    f"[shard {args.shard}] pass {pass_idx} | "
                    f"items_done={writer.items_done} sft={writer.sft_items} "
                    f"zero={writer.zero_hit_items} ({rate:.2f} items/s) | "
                    f"{gpu_util_string()}",
                    flush=True,
                )
        flush_batch()
        if not args.no_stop:
            break
        if n_seen_this_pass == 0:
            print(f"[shard {args.shard}] no work left this pass, sleeping 30s", flush=True)
            time.sleep(30)
        # Re-load skip set for next pass (other shards may have written in
        # parallel — though hash-mod guarantees no overlap, this also picks up
        # any zero-hits we just recorded).
        skip = writer.load_skip_set() if args.resume else set()

    elapsed = time.time() - t0
    print(
        f"[done] shard {args.shard}: {writer.items_done} items, "
        f"{writer.sft_items} sft, {writer.zero_hit_items} zero-hits "
        f"in {elapsed/60:.1f} min. Dir: {writer.shard_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
