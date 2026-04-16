"""
`anaisleila/computer-use-data-psai` — real screenshots + weak supervision from event logs.

Labels are derived from the first ``click`` in ``events`` (normalized by ``screen_width`` /
``screen_height`` from ``metadata``) and hashed text fields for auxiliary heads so the
existing multi-task loss remains well-defined.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelConfig, grid_cells
from .dataset import task_text_to_ids


def _stable_seed(s: str) -> int:
    h = 2166136261
    for c in s.encode("utf-8", errors="ignore"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _thought_ids_from_seed(seed: int, cfg: ModelConfig) -> List[int]:
    s = seed & 0xFFFFFFFF
    out: List[int] = []
    for _ in range(cfg.thought_len):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(s % cfg.thought_vocab_size)
    return out


def _action_name_to_id(name: str, cfg: ModelConfig) -> int:
    n = (name or "").lower().strip()
    order = (
        "click",
        "scroll",
        "key",
        "write",
        "press",
        "move",
        "drag",
    )
    if n in order:
        return order.index(n) % cfg.num_action_types
    return _stable_seed(n) % cfg.num_action_types


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _first_click(events: Any) -> Optional[Dict[str, Any]]:
    if isinstance(events, str) and events.strip():
        try:
            events = json.loads(events)
        except json.JSONDecodeError:
            return None
    if not isinstance(events, list):
        return None
    for e in events:
        if not isinstance(e, dict):
            continue
        if str(e.get("action", "")).lower() == "click":
            return e
    return None


def _pil_to_tensor(img: Any, image_size: int) -> torch.Tensor:
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(img)}")
    rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def row_to_sample(row: Dict[str, Any], cfg: ModelConfig) -> Optional[Dict[str, Any]]:
    shots = row.get("screenshots") or []
    if not shots:
        return None
    img0 = shots[0]
    click = _first_click(row.get("events"))
    if click is None:
        return None

    md = _parse_metadata(row.get("metadata"))
    sw = float(md.get("screen_width") or 0.0)
    sh = float(md.get("screen_height") or 0.0)
    if sw <= 1.0 or sh <= 1.0:
        from PIL import Image

        if isinstance(img0, Image.Image):
            sw, sh = float(img0.size[0]), float(img0.size[1])
        else:
            return None

    x = float(click.get("x", sw * 0.5))
    y = float(click.get("y", sh * 0.5))
    xn = max(0.0, min(1.0, x / sw))
    yn = max(0.0, min(1.0, y / sh))

    g = grid_cells(cfg)
    col = min(cfg.grid_w - 1, max(0, int(xn * cfg.grid_w)))
    row_i = min(cfg.grid_h - 1, max(0, int(yn * cfg.grid_h)))
    click_cell = row_i * cfg.grid_w + col

    task = str(row.get("task_name") or row.get("taskId") or "task")
    task_ids = task_text_to_ids(task, cfg.task_max_len, cfg.task_vocab_size)

    seed = _stable_seed(str(row.get("unique_data_id") or task))
    thought = _thought_ids_from_seed(seed, cfg)
    obj_t = seed % cfg.num_object_types

    bx = min(cfg.bbox_bins - 1, max(0, int(xn * cfg.bbox_bins)))
    by = min(cfg.bbox_bins - 1, max(0, int(yn * cfg.bbox_bins)))
    bw = max(1, min(cfg.bbox_bins - 1, cfg.bbox_bins // 8))
    bh = max(1, min(cfg.bbox_bins - 1, cfg.bbox_bins // 8))

    a_type = _action_name_to_id("click", cfg)
    key_id = (seed >> 8) % cfg.num_key_ids

    try:
        image = _pil_to_tensor(img0, cfg.image_size)
    except Exception:
        return None

    return {
        "image": image,
        "task_token_ids": task_ids,
        "thought_ids": torch.tensor(thought, dtype=torch.long),
        "object_type": torch.tensor(obj_t, dtype=torch.long),
        "bbox_bins": torch.tensor([bx, by, bw, bh], dtype=torch.long),
        "action_type": torch.tensor(a_type, dtype=torch.long),
        "click_cell": torch.tensor(click_cell, dtype=torch.long),
        "key_id": torch.tensor(key_id, dtype=torch.long),
        "task_text": task,
    }


_PSAI_COLUMNS = (
    "unique_data_id",
    "task_name",
    "screenshots",
    "events",
    "metadata",
)


def _stream_train() -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        "anaisleila/computer-use-data-psai",
        split="train",
        streaming=True,
        columns=list(_PSAI_COLUMNS),
    )
    yield from ds


class PSAIComputerUseDataset(Dataset):
    """
    Buffer ``n`` usable rows from the PSAI train split (streaming), after skipping the first
    ``skip`` **raw** rows (not necessarily valid). If too few valid rows are found, the
    dataset may be shorter than ``n``.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        n: int,
        skip: int = 0,
        *,
        max_raw_scans: int = 200_000,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.samples: List[Dict[str, Any]] = []
        it = _stream_train()
        skipped = 0
        scanned = 0
        while skipped < skip and scanned < max_raw_scans:
            try:
                next(it)
            except StopIteration:
                break
            skipped += 1
            scanned += 1
        while len(self.samples) < n and scanned < max_raw_scans:
            try:
                row = next(it)
            except StopIteration:
                break
            scanned += 1
            s = row_to_sample(row, cfg)
            if s is not None:
                self.samples.append(s)
                if len(self.samples) % max(1, min(8, n)) == 0 or len(self.samples) == n:
                    print(
                        f"[psai] buffered {len(self.samples)}/{n} usable rows "
                        f"(raw scanned {scanned})",
                        file=sys.stderr,
                        flush=True,
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
