"""DEPRECATED shim. The Northstar action parser now lives in the repo-root
``northstar_format`` module so the sample-generation and eval pipelines share
one source of truth. This file re-exports the legacy ``parse_click(raw) ->
(x, y)``, ``to_pixel``, ``hit`` API for any straggling imports.

Prefer importing from ``northstar_format`` directly:

    from northstar_format import parse_action, denormalize, hit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from northstar_format import (  # noqa: E402,F401
    COORD_SPACE,
    denormalize,
    parse_action,
)

__all__ = ["parse_click", "to_pixel", "hit", "COORD_SPACE"]


def parse_click(raw_response: str) -> Optional[Tuple[float, float]]:
    """Legacy adapter: returns just (x, y) in 0-1000 normalized space, or None.

    New code should call ``northstar_format.parse_action`` directly to get the
    full ``{type, x, y}`` dict.
    """
    parsed = parse_action(raw_response)
    if parsed is None:
        return None
    return float(parsed["x"]), float(parsed["y"])


def to_pixel(point_norm: Tuple[float, float], img_w: int, img_h: int) -> Tuple[float, float]:
    """Legacy adapter: 0-1000 normalized point -> pixel coords (floats)."""
    x, y = point_norm
    px, py = denormalize(int(round(x)), int(round(y)), img_w, img_h)
    return float(px), float(py)


def hit(point_pixel: Tuple[float, float], bbox_xyxy) -> bool:
    """Inclusive containment check; bbox is [x1, y1, x2, y2] in pixels."""
    x, y = point_pixel
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2
