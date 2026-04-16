#!/usr/bin/env python3
"""
Probe Hugging Face computer-use / GUI datasets and print schema summaries.

Does not download large optional artifacts (PSAI videos). Set HF_TOKEN for
reliable Hub access.

Usage:
  python3 prototypes/gui_neurosymbolic/dataset_sources/compile_hf_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List


def _preview(v: Any, max_len: int = 160) -> str:
    s = repr(v)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def probe_open_computer_using_agent() -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset(
        "arthurcolle/open-computer-using-agent",
        split="train",
        streaming=True,
    )
    row = next(iter(ds.take(1)))
    out: Dict[str, Any] = {"keys": list(row.keys())}
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        m0 = msgs[0]
        if isinstance(m0, dict):
            out["first_message_keys"] = list(m0.keys())
            c = m0.get("content")
            out["first_content_type"] = type(c).__name__
            out["first_content_preview"] = _preview(c, 240)
    return out


def probe_agentnet() -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("xlangai/AgentNet", split="train", streaming=True)
    row = next(iter(ds.take(1)))
    keys = sorted(row.keys())
    sample: Dict[str, str] = {}
    for k in keys:
        sample[k] = f"{type(row[k]).__name__}: {_preview(row[k], 120)}"
    return {"keys": keys, "fields": sample}


def probe_pc_agent_e_arrow() -> Dict[str, Any]:
    """Inspect raw Image storage; Hub revision may point jsonl paths instead of PNG bytes."""
    from datasets import load_dataset

    ds = load_dataset(
        "henryhe0123/PC-Agent-E",
        split="train[:1]",
        verification_mode="no_checks",
    )
    tbl = ds.data.table
    col = tbl.column("image")
    v0 = col[0].as_py()
    if isinstance(v0, dict):
        b = v0.get("bytes")
        p = v0.get("path") or ""
        inner = p
        if "::" in p:
            inner = p.split("::", 1)[0]
        if inner.startswith("zip://"):
            inner = inner[len("zip://") :]
        return {
            "arrow_image_struct": True,
            "bytes_is_none": b is None,
            "bytes_len": None if b is None else len(b),
            "path_inside_zip_preview": inner[:200],
        }
    return {"arrow_image_struct": False, "value_preview": _preview(v0)}


def probe_psai_skip() -> Dict[str, Any]:
    return {
        "note": "Skipped automatic probe (large download). Use load_dataset('anaisleila/computer-use-data-psai') and dedupe on unique_data_id.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=("open", "agentnet", "pc-agent-e", "psai", "all"),
        default="all",
    )
    args = p.parse_args()

    sections: List[Callable[[], Dict[str, Any]]] = []
    labels: List[str] = []

    def add(label: str, fn: Callable[[], Dict[str, Any]]) -> None:
        labels.append(label)
        sections.append(fn)

    if args.only in ("all", "open"):
        add("arthurcolle/open-computer-using-agent", probe_open_computer_using_agent)
    if args.only in ("all", "agentnet"):
        add("xlangai/AgentNet", probe_agentnet)
    if args.only in ("all", "pc-agent-e"):
        add("henryhe0123/PC-Agent-E (Arrow image column)", probe_pc_agent_e_arrow)
    if args.only in ("all", "psai"):
        add("anaisleila/computer-use-data-psai", probe_psai_skip)

    report: Dict[str, Any] = {}
    for label, fn in zip(labels, sections):
        try:
            report[label] = {"ok": True, "data": fn()}
        except Exception as e:
            report[label] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
