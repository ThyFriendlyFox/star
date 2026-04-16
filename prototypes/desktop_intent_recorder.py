#!/usr/bin/env python3
"""
Record timed desktop screenshots plus low-level input events, then (optionally) call an
LLM to summarize inferred user intent per time segment.

**macOS:** grant Screen Recording and Accessibility to your terminal / Python in
System Settings → Privacy & Security.

**Security:** keyboard logging is off by default. Enable only on a machine you control
and avoid typing secrets while ``--keyboard`` is on.

Examples::

  # Record primary monitor, 2s between frames, mouse only, max 10 minutes
  python3 prototypes/desktop_intent_recorder.py record --out out/screen_sessions --duration-sec 600 --interval-sec 2

  # Include key names (not recommended on shared machines)
  python3 prototypes/desktop_intent_recorder.py record --out out/screen_sessions --keyboard

  # After recording, annotate with OpenAI (export OPENAI_API_KEY)
  python3 prototypes/desktop_intent_recorder.py annotate --session out/screen_sessions/session_20260416_120000

  # Local vision model via Ollama OpenAI-compatible API
  python3 prototypes/desktop_intent_recorder.py annotate --session ... --base-url http://127.0.0.1:11434/v1 --model llava

Dependencies (see requirements_desktop.txt)::

  pip install -r requirements_desktop.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(out: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    d = out / f"session_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "screenshots").mkdir(exist_ok=True)
    return d


def _write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def run_record(args: argparse.Namespace) -> None:
    try:
        import mss
        import mss.tools
    except ImportError as e:
        raise SystemExit("Install recording deps: pip install -r requirements_desktop.txt") from e

    out = Path(args.out).expanduser().resolve()
    session = _session_dir(out)
    events_path = session / "events.jsonl"
    meta: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "interval_sec": float(args.interval_sec),
        "duration_sec": float(args.duration_sec) if args.duration_sec else None,
        "keyboard": bool(args.keyboard),
        "platform": sys.platform,
        "session_dir": str(session),
    }
    with open(session / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    stop = threading.Event()
    frame_idx = {"n": 0}
    lock = threading.Lock()

    def shot_loop() -> None:
        with mss.mss() as sct:
            mon = sct.monitors[int(args.monitor)]
            deadline = None
            if args.duration_sec:
                deadline = time.time() + float(args.duration_sec)
            while not stop.is_set():
                if deadline is not None and time.time() >= deadline:
                    stop.set()
                    break
                t0 = time.time()
                try:
                    grab = sct.grab(mon)
                    idx = frame_idx["n"]
                    png_name = f"{idx:06d}.png"
                    png_path = session / "screenshots" / png_name
                    mss.tools.to_png(grab.rgb, grab.size, output=str(png_path))
                    with lock:
                        frame_idx["n"] += 1
                    _write_jsonl(
                        events_path,
                        {
                            "t": _utc_now_iso(),
                            "type": "screenshot",
                            "index": idx,
                            "path": f"screenshots/{png_name}",
                            "size": [grab.width, grab.height],
                        },
                    )
                except Exception as e:
                    _write_jsonl(
                        events_path,
                        {"t": _utc_now_iso(), "type": "error", "where": "screenshot", "message": str(e)},
                    )
                dt = time.time() - t0
                wait = max(0.05, float(args.interval_sec) - dt)
                if stop.wait(timeout=wait):
                    break

    def on_move(x: float, y: float) -> None:
        if args.mouse_move:
            _write_jsonl(events_path, {"t": _utc_now_iso(), "type": "mouse_move", "x": x, "y": y})

    def on_click(x: float, y: float, button: Any, pressed: bool) -> None:
        _write_jsonl(
            events_path,
            {
                "t": _utc_now_iso(),
                "type": "mouse_click",
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed,
            },
        )

    def on_scroll(x: float, y: float, dx: float, dy: float) -> None:
        _write_jsonl(
            events_path,
            {"t": _utc_now_iso(), "type": "scroll", "x": x, "y": y, "dx": dx, "dy": dy},
        )

    def on_press(key: Any) -> None:
        if not args.keyboard:
            return
        try:
            name = key.char if hasattr(key, "char") and key.char is not None else str(key)
        except AttributeError:
            name = str(key)
        _write_jsonl(events_path, {"t": _utc_now_iso(), "type": "key_press", "key": name})

    t_shot = threading.Thread(target=shot_loop, daemon=True)
    t_shot.start()

    listeners = []
    try:
        from pynput import mouse, keyboard

        ml = mouse.Listener(on_move=on_move if args.mouse_move else None, on_click=on_click, on_scroll=on_scroll)
        ml.start()
        listeners.append(ml)
        if args.keyboard:
            kl = keyboard.Listener(on_press=on_press)
            kl.start()
            listeners.append(kl)
    except ImportError as e:
        print("Warning: pynput not installed; only screenshots will be recorded.", file=sys.stderr)
        print(e, file=sys.stderr)

    def _stop(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    print(f"Recording to {session}", flush=True)
    print("Ctrl+C to stop.", flush=True)
    while not stop.is_set():
        time.sleep(0.2)
    t_shot.join(timeout=5.0)
    for L in listeners:
        try:
            L.stop()
        except Exception:
            pass
    meta["ended_utc"] = _utc_now_iso()
    meta["frames"] = frame_idx["n"]
    with open(session / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {frame_idx['n']} frames under {session}", flush=True)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _load_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _screenshot_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("type") == "screenshot"]


def _events_text_slice(events: List[Dict[str, Any]], max_lines: int = 80) -> str:
    lines: List[str] = []
    for e in events:
        if e.get("type") == "screenshot":
            continue
        lines.append(json.dumps(e, ensure_ascii=False))
        if len(lines) >= max_lines:
            lines.append("… (truncated)")
            break
    return "\n".join(lines) if lines else "(no pointer/keyboard events in this slice)"


def _png_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _parse_t(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _build_time_segments(
    shots: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    segment_sec: float,
    max_img: int,
) -> List[tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Return list of (t_start, t_end, screenshot_rows, event_rows_in_range)."""
    if not shots:
        return []
    t_first = _parse_t(shots[0]["t"])
    t_last = _parse_t(shots[-1]["t"])
    out: List[tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    cur = t_first
    while cur <= t_last + 1e-6:
        t_end = cur + segment_sec
        chunk_shots = [s for s in shots if cur <= _parse_t(s["t"]) < t_end]
        if len(chunk_shots) > max_img:
            chunk_shots = chunk_shots[-max_img:]
        chunk_ev = [e for e in events if cur <= _parse_t(e["t"]) < t_end]
        if chunk_shots:
            out.append((cur, t_end, chunk_shots, chunk_ev))
        cur = t_end
    if not out:
        # single bucket with all data
        chunk_shots = shots[-max_img:] if len(shots) > max_img else shots
        chunk_ev = list(events)
        out.append((t_first, t_last + 1.0, chunk_shots, chunk_ev))
    return out


def run_annotate(args: argparse.Namespace) -> None:
    session = Path(args.session).expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"Not a directory: {session}")
    events_path = session / "events.jsonl"
    events = _load_events(events_path)
    shots = _screenshot_rows(events)
    if not shots:
        raise SystemExit("No screenshot events in events.jsonl")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai (see requirements_desktop.txt)") from e

    api_key = args.api_key or __import__("os").environ.get("OPENAI_API_KEY", "")
    if not args.base_url and not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key, or use --base-url for a local server.")

    client = OpenAI(api_key=api_key or "ollama", base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)

    max_img = int(args.max_images_per_segment)
    segments = _build_time_segments(shots, events, float(args.segment_sec), max_img)

    out_lines: List[Dict[str, Any]] = []

    for i, (ta, tb, shot_chunk, chunk_ev) in enumerate(segments):
        ev_txt = _events_text_slice(
            [e for e in chunk_ev if e.get("type") != "screenshot"][:120]
        )

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are given sampled desktop screenshots and a machine-generated log of "
                    "mouse/keyboard activity for one continuous segment. Infer the user's likely "
                    "high-level intent: what task they are trying to accomplish and in which "
                    "application context. Reply with 2-4 concise sentences. If uncertain, say so.\n\n"
                    f"Activity log (JSON lines, may be partial):\n{ev_txt}"
                ),
            }
        ]
        for s in shot_chunk:
            p = session / s["path"]
            if not p.is_file():
                continue
            b64 = _png_b64(p)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        n_img = sum(1 for c in content if c.get("type") == "image_url")
        if n_img == 0:
            rec = {
                "segment_index": i,
                "t_start": datetime.fromtimestamp(ta, tz=timezone.utc).isoformat(),
                "t_end": datetime.fromtimestamp(tb, tz=timezone.utc).isoformat(),
                "model": args.model,
                "intent": "[skipped: no screenshot files found on disk for this segment]",
            }
            out_lines.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            continue

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            text = f"[error calling model: {e}]"

        rec = {
            "segment_index": i,
            "t_start": datetime.fromtimestamp(ta, tz=timezone.utc).isoformat(),
            "t_end": datetime.fromtimestamp(tb, tz=timezone.utc).isoformat(),
            "model": args.model,
            "intent": text,
        }
        out_lines.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    out_path = session / "intents.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_lines)} segments to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Capture screenshots and input events")
    pr.add_argument("--out", type=str, default="out/screen_sessions", help="Base directory for sessions")
    pr.add_argument("--interval-sec", type=float, default=3.0, help="Time between screenshots")
    pr.add_argument("--duration-sec", type=float, default=0.0, help="Stop after N seconds (0 = until Ctrl+C)")
    pr.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary on most setups)")
    pr.add_argument("--mouse-move", action="store_true", help="Log every mouse move (very verbose)")
    pr.add_argument("--keyboard", action="store_true", help="Log key presses (avoid on untrusted/shared use)")
    pr.set_defaults(_run=run_record)

    pa = sub.add_parser("annotate", help="LLM intent backfill for a session folder")
    pa.add_argument("--session", type=str, required=True, help="Path to session_* directory")
    pa.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model with vision")
    pa.add_argument("--segment-sec", type=float, default=45.0, help="Target wall-clock span per LLM call")
    pa.add_argument("--max-images-per-segment", type=int, default=6, help="Cap images sent per segment")
    pa.add_argument("--max-tokens", type=int, default=400)
    pa.add_argument("--temperature", type=float, default=0.3)
    pa.add_argument("--api-key", type=str, default="", help="Override OPENAI_API_KEY")
    pa.add_argument(
        "--base-url",
        type=str,
        default="",
        help="OpenAI-compatible API base (e.g. http://127.0.0.1:11434/v1 for Ollama)",
    )
    pa.set_defaults(_run=run_annotate)

    args = p.parse_args()
    args._run(args)


if __name__ == "__main__":
    main()
