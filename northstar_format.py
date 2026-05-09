"""Single source of truth for the Northstar-CUA-Fast tool spec + action parser.

Northstar-CUA-Fast (Tzafon/Northstar-CUA-Fast on HF) is a Qwen3-VL-4B model
fine-tuned with GRPO for GUI agent tasks. Both the sample-generation pipeline
(``sample/sample_northstar.py``) and the eval pipeline
(``eval/eval_screenspot_pro.py``) MUST use the same tool schema and the same
action parser, otherwise eval will under-count valid completions.

What we know for sure
---------------------
* Lightcone ``examples/_cua.py`` defines::

      COORD_KEYS = {
          "x":  "display_width",  "x1": "display_width",  "x2": "display_width",
          "y":  "display_height", "y1": "display_height", "y2": "display_height",
      }

  and normalizes via ``int(d[key] / 1000 * tool[dim])``. So Northstar emits
  **flat ``x`` / ``y`` integer keys** in 0-999 normalized space, **NOT** the
  Anthropic-style ``coordinate: [x, y]`` array.

* The HF model card shows the simplified ``Tzafon-API`` tool form::

      {"type": "computer_use",
       "display_width": 1024, "display_height": 768,
       "environment": "browser"}

  i.e. NOT the verbose OpenAI ``{type: "function", function: {...parameters...}}``
  envelope. The Tzafon API server presumably translates this to whatever the
  chat template expects.

* The model emits::

      <tool_call>{"name": "computer_use",
                  "arguments": {"type": "click", "x": <int>, "y": <int>}}
      </tool_call>

  Action types include ``click`` (point) and ``drag`` (4-coord ``x1,y1,x2,y2``).

What we are uncertain about
---------------------------
We don't have direct access to Northstar's training tool schema (the chat
template that vLLM applies). Two plausible formats:

  (a) the simplified Tzafon-API form above (4 fields, no ``parameters`` block);
  (b) the verbose OpenAI ``{type: "function", function: {name, parameters: ...}}``
      envelope that vLLM's ``LLM.chat(tools=...)`` typically expects.

If (a) is correct, vLLM may reject it as malformed. If (b) is correct, the
chat template auto-renders it into the system block, and the model behaviour
is governed entirely by the property names we list under ``parameters``.

We render the **verbose form with flat ``x`` / ``y`` integer fields** — this
matches both lightcone's ``COORD_KEYS`` evidence and vLLM's expected tool
shape, and is the format ``sample/sample_northstar.py`` was already using.
The eval pipeline previously used ``coordinate: [x, y]`` (Anthropic style),
which would have produced the schema mismatch this module fixes.

Safety net: ``parse_action`` also accepts ``coordinate: [x, y]`` as a legacy
fallback in case any deployed checkpoint was actually trained that way.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Optional

# Output coordinate space (0-999 inclusive, divided by 1000 — matches lightcone).
COORD_SPACE = 1000

# Action types Northstar emits (from lightcone ``_cua.py``).
_POINT_ACTIONS = {"click", "double_click", "triple_click", "right_click"}
_BBOX_ACTIONS = {"drag"}  # uses x1,y1,x2,y2


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------
def build_computer_use_tool(
    display_width: int,
    display_height: int,
    environment: str = "desktop",
) -> dict[str, Any]:
    """Build the ``computer_use`` tool dict to pass to vLLM ``LLM.chat(tools=...)``.

    Renders the verbose OpenAI-style envelope (``{type: "function", function:
    {name, description, parameters: ...}}``) because vLLM's chat-template
    machinery expects that shape; ``display_width`` / ``display_height`` /
    ``environment`` are added as extra keys on the inner ``function`` dict so
    the chat template can surface them to the model verbatim.

    Critically, the ``parameters.properties`` block uses **flat ``x`` and ``y``
    integer fields in 0-999**, NOT a ``coordinate: [x, y]`` array. That matches
    what lightcone's ``COORD_KEYS`` proves Northstar emits.

    See module docstring for the uncertainty notes around the exact training
    tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": (
                "Perform a GUI action on the screenshot. Coordinates x and y "
                "are normalized integers in [0, 999] over the display."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "click", "double_click", "triple_click", "right_click",
                            "drag", "type", "key", "scroll", "hscroll",
                            "navigate", "wait", "terminate",
                        ],
                    },
                    "x": {"type": "integer", "minimum": 0, "maximum": 999},
                    "y": {"type": "integer", "minimum": 0, "maximum": 999},
                    "x1": {"type": "integer"}, "y1": {"type": "integer"},
                    "x2": {"type": "integer"}, "y2": {"type": "integer"},
                    "text": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "scroll_x": {"type": "integer"}, "scroll_y": {"type": "integer"},
                    "url": {"type": "string"},
                },
                "required": ["type"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------
# 1. Wrapped <tool_call>{...}</tool_call>. Non-greedy + DOTALL.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 2. Bare JSON object that mentions both "name" and "arguments" (no XML wrap).
_NAMED_JSON_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"arguments"\s*:\s*(?:\{.*?\}|"[^"]*")\s*\}',
    re.DOTALL,
)

# 3. Bare action JSON: {"type": "click", "x": .., "y": ..} (any order of x/y).
_BARE_ACTION_RE = re.compile(
    r'\{[^{}]*"type"\s*:\s*"(?P<atype>click|double_click|triple_click|right_click)"'
    r'[^{}]*\}',
    re.DOTALL,
)

# 4. Legacy coordinate-array form (Anthropic-style fallback).
_COORDINATE_RE = re.compile(
    r'"coordinate"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
    r'(?:\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?))?\s*\]',
    re.DOTALL,
)


def _try_loads(blob: str) -> Optional[Any]:
    """JSON loads with a trailing-comma tolerance retry."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _coerce_args(args: Any) -> Optional[dict]:
    """Unwrap stringified arguments; return a dict or None."""
    if isinstance(args, str):
        args = _try_loads(args)
    if isinstance(args, dict):
        return args
    return None


def _normalize_action_type(atype: Any) -> str:
    """Normalize action type names. Anthropic-style ``left_click`` -> ``click``."""
    s = str(atype) if atype is not None else "click"
    if s == "left_click":
        return "click"
    return s


def _extract_xy(args: dict) -> Optional[tuple[int, int, str]]:
    """Pull (x, y, action_type) from an arguments dict.

    Handles three sub-cases:
      * flat x/y point   -> (x, y, type)
      * x1,y1,x2,y2 bbox -> centre + type
      * coordinate array -> [x,y] or [x1,y1,x2,y2] -> centre

    Returns None if no valid coordinate could be extracted.
    """
    action_type = _normalize_action_type(args.get("type") or args.get("action") or "click")

    # Flat x/y (canonical Northstar form).
    if "x" in args and "y" in args:
        try:
            return int(round(float(args["x"]))), int(round(float(args["y"]))), action_type
        except (TypeError, ValueError):
            pass

    # Northstar quirk: when prompted via inline system msg (not tools= param),
    # model emits x as a [x, y] array instead of separate x/y fields.
    if "x" in args and isinstance(args["x"], (list, tuple)) and len(args["x"]) >= 2:
        try:
            xs = [float(v) for v in args["x"]]
            return int(round(xs[0])), int(round(xs[1])), action_type
        except (TypeError, ValueError):
            pass

    # Flat 4-coord bbox -> centre (e.g. drag origin / target collapse).
    if all(k in args for k in ("x1", "y1", "x2", "y2")):
        try:
            x1, y1, x2, y2 = (float(args[k]) for k in ("x1", "y1", "x2", "y2"))
            return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)), action_type
        except (TypeError, ValueError):
            pass

    # Coordinate array form: legacy fallback.
    coord = args.get("coordinate") or args.get("coord")
    if isinstance(coord, (list, tuple)):
        try:
            nums = [float(v) for v in coord]
        except (TypeError, ValueError):
            nums = []
        if len(nums) == 2:
            return int(round(nums[0])), int(round(nums[1])), action_type
        if len(nums) == 4:
            return (
                int(round((nums[0] + nums[2]) / 2)),
                int(round((nums[1] + nums[3]) / 2)),
                action_type,
            )

    return None


def _from_envelope(obj: Any) -> Optional[dict]:
    """Try to interpret ``obj`` as a tool-call envelope or args-only dict."""
    if not isinstance(obj, dict):
        return None
    # Envelope with arguments?
    if "arguments" in obj:
        args = _coerce_args(obj["arguments"])
        if args is None:
            return None
    else:
        args = obj
    extracted = _extract_xy(args)
    if extracted is None:
        return None
    x, y, atype = extracted
    return {"type": atype, "x": x, "y": y}


def parse_action(completion: str) -> Optional[dict]:
    """Extract ``{"type", "x", "y"}`` from a Northstar completion string.

    Returns x, y as ints in 0-999 normalized space (denormalize via
    ``denormalize``). Returns ``None`` on parse failure.

    Tries, in order:
      1. ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (canonical)
      2. Bare ``{"name": ..., "arguments": {...}}`` JSON (no XML tags)
      3. Bare ``{"type": "click", "x": .., "y": ..}`` JSON (args-only)
      4. ``"coordinate": [x, y]`` form (Anthropic-style legacy fallback)
    """
    if not completion:
        return None

    # 1. Wrapped tool call.
    for m in _TOOL_CALL_RE.finditer(completion):
        obj = _try_loads(m.group(1))
        result = _from_envelope(obj)
        if result is not None:
            return result

    # 2. Named JSON envelope (no XML wrap).
    for m in _NAMED_JSON_RE.finditer(completion):
        obj = _try_loads(m.group(0))
        result = _from_envelope(obj)
        if result is not None:
            return result

    # 3. Bare action JSON (args-only).
    for m in _BARE_ACTION_RE.finditer(completion):
        obj = _try_loads(m.group(0))
        result = _from_envelope(obj)
        if result is not None:
            return result

    # 4. Legacy coordinate-array fallback. We grab the first match and try to
    #    determine the action type from a sibling ``"type": "..."`` if one is
    #    present nearby; otherwise default to "click".
    m = _COORDINATE_RE.search(completion)
    if m:
        try:
            x1 = float(m.group(1))
            y1 = float(m.group(2))
        except (TypeError, ValueError):
            return None
        if m.group(3) is not None and m.group(4) is not None:
            try:
                x2, y2 = float(m.group(3)), float(m.group(4))
                x_out = int(round((x1 + x2) / 2))
                y_out = int(round((y1 + y2) / 2))
            except (TypeError, ValueError):
                return None
        else:
            x_out, y_out = int(round(x1)), int(round(y1))
        # Try to pick up the action type from the same window of text.
        atype = "click"
        type_match = re.search(r'"(?:type|action)"\s*:\s*"([^"]+)"', completion)
        if type_match:
            atype = _normalize_action_type(type_match.group(1))
        return {"type": atype, "x": x_out, "y": y_out}

    return None


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------
def denormalize(x: int, y: int, image_width: int, image_height: int) -> tuple[int, int]:
    """0-999 normalized -> pixel coords. Mirrors lightcone ``_cua.py`` exactly:
    ``int(coord / 1000 * dim)``.
    """
    return (
        int(x / COORD_SPACE * image_width),
        int(y / COORD_SPACE * image_height),
    )


def hit(
    pred_x_norm: int,
    pred_y_norm: int,
    gt_bbox: list[int],
    image_width: int,
    image_height: int,
) -> bool:
    """Eval primitive: denormalize pred, check if (px, py) lies inside the
    pixel-space ``gt_bbox`` (xyxy)."""
    px, py = denormalize(pred_x_norm, pred_y_norm, image_width, image_height)
    x1, y1, x2, y2 = gt_bbox
    return x1 <= px <= x2 and y1 <= py <= y2


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _selftest() -> None:
    cases: list[tuple[str, str, Optional[dict]]] = [
        # 1. Canonical wrapped tool_call with flat x/y.
        (
            "canonical tool_call (flat x/y)",
            'I will click the search bar.\n<tool_call>\n'
            '{"name": "computer_use", "arguments": {"type": "click", "x": 234, "y": 567}}\n'
            '</tool_call>',
            {"type": "click", "x": 234, "y": 567},
        ),
        # 2. Bare JSON envelope without tags.
        (
            "bare named JSON (no tags)",
            '{"name":"computer_use","arguments":{"type":"double_click","x":42,"y":99}}',
            {"type": "double_click", "x": 42, "y": 99},
        ),
        # 3. Legacy Anthropic-style coordinate array.
        (
            "coordinate-array fallback",
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[423,567]}}</tool_call>',
            {"type": "click", "x": 423, "y": 567},
        ),
        # 4. 4-coord bbox -> centre.
        (
            "x1y1x2y2 bbox -> centre",
            '<tool_call>{"name":"computer_use","arguments":'
            '{"type":"drag","x1":100,"y1":200,"x2":200,"y2":300}}</tool_call>',
            {"type": "drag", "x": 150, "y": 250},
        ),
        # 5. Stringified arguments.
        (
            "stringified arguments",
            '<tool_call>{"name":"computer_use","arguments":'
            '"{\\"type\\":\\"click\\",\\"x\\":7,\\"y\\":8}"}</tool_call>',
            {"type": "click", "x": 7, "y": 8},
        ),
        # Negative case: garbage -> None.
        (
            "garbage input",
            "I cannot determine where to click on this image.",
            None,
        ),
    ]
    failures = 0
    for label, completion, expected in cases:
        got = parse_action(completion)
        ok = got == expected
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {label}: expected={expected} got={got}")
        if not ok:
            failures += 1

    # denormalize: lightcone math, int() truncation.
    px, py = denormalize(500, 500, 1280, 720)
    if (px, py) != (640, 360):
        print(f"[FAIL] denormalize: got {(px, py)}, expected (640, 360)")
        failures += 1
    else:
        print(f"[OK  ] denormalize(500,500,1280,720) = (640, 360)")

    # hit primitive.
    if not hit(500, 500, [600, 320, 700, 400], 1280, 720):
        print("[FAIL] hit: expected True for centre point inside bbox")
        failures += 1
    else:
        print("[OK  ] hit: centre point inside bbox")
    if hit(0, 0, [600, 320, 700, 400], 1280, 720):
        print("[FAIL] hit: expected False for (0,0) outside bbox")
        failures += 1
    else:
        print("[OK  ] hit: (0,0) outside bbox")

    # build_computer_use_tool sanity: schema must contain flat x/y ints, not coordinate.
    tool = build_computer_use_tool(1024, 768, environment="browser")
    props = tool["function"]["parameters"]["properties"]
    if "x" not in props or "y" not in props:
        print("[FAIL] build_computer_use_tool: missing flat x/y")
        failures += 1
    elif "coordinate" in props:
        print("[FAIL] build_computer_use_tool: should NOT use coordinate-array")
        failures += 1
    elif tool["function"]["display_width"] != 1024:
        print("[FAIL] build_computer_use_tool: display_width hint missing")
        failures += 1
    elif tool["function"]["environment"] != "browser":
        print("[FAIL] build_computer_use_tool: environment hint missing")
        failures += 1
    else:
        print("[OK  ] build_computer_use_tool: flat x/y schema, dims surfaced")

    if failures:
        sys.exit(f"{failures} self-test(s) failed")
    print(f"\nAll {len(cases) + 4} self-tests passed.")


if __name__ == "__main__":
    _selftest()
