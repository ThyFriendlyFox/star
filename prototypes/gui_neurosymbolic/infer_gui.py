#!/usr/bin/env python3
"""
Simple desktop UI: pick an image, enter an instruction, run the GUI model, and show
bounding box + action label overlaid on the **square model input** (same resize as inference).

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/infer_gui.py \\
        --checkpoint out/gui_neurosymbolic_psai_full.pt
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk

from prototypes.gui_neurosymbolic.config import ModelConfig
from prototypes.gui_neurosymbolic.dataset import task_text_to_ids
from prototypes.gui_neurosymbolic.infer import load_image
from prototypes.gui_neurosymbolic.model import build_model
from prototypes.gui_neurosymbolic.symbolic_planner import neural_logits_to_action_dict


def _bbox_norm_to_xyxy(
    bbox_norm: List[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    cx, cy, bw, bh = bbox_norm
    half_w = bw / 2.0
    half_h = bh / 2.0
    x0 = (cx - half_w) * width
    y0 = (cy - half_h) * height
    x1 = (cx + half_w) * width
    y1 = (cy + half_h) * height
    return x0, y0, x1, y1


def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """[3,H,W] float 0..1 -> RGB PIL."""
    x = img_t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _action_caption(action: Dict[str, Any]) -> str:
    t = str(action.get("type", "?"))
    if t == "click":
        return f"click  x={action.get('x', 0):.3f}  y={action.get('y', 0):.3f}"
    if t == "keypress":
        return f"key  id={action.get('key_id', '?')}"
    return t


def _draw_overlay(
    base_rgb: Image.Image,
    symbols: Dict[str, Any],
    action: Dict[str, Any],
) -> Image.Image:
    w, h = base_rgb.size
    out = base_rgb.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    bn = symbols.get("bbox_norm")
    if isinstance(bn, list) and len(bn) == 4:
        x0, y0, x1, y1 = _bbox_norm_to_xyxy([float(x) for x in bn], w, h)
        pad = 2
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline="#00c853", width=3)

    t = str(action.get("type", ""))
    if t == "click":
        cx = float(action.get("x", 0.5)) * w
        cy = float(action.get("y", 0.5)) * h
        r = max(4, min(w, h) // 80)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#d50000", width=3)

    oid = symbols.get("object_type_id", "?")
    cap = _action_caption(action)
    line1 = f"{cap}   |   object {oid}"
    bbox = draw.textbbox((0, 0), line1, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 6
    draw.rectangle([0, 0, w, th + 2 * margin], fill=(0, 0, 0))
    draw.text((margin, margin), line1, fill="#ffffff", font=font)

    return out


def _fit_display(pil: Image.Image, max_side: int = 900) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    s = max_side / float(m)
    nw, nh = int(w * s), int(h * s)
    return pil.resize((nw, nh), Image.LANCZOS)


class InferGuiApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        checkpoint: Path,
        device: str = "cpu",
    ) -> None:
        self.root = root
        self.checkpoint = checkpoint
        self.device_s = device
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.cfg: Optional[ModelConfig] = None
        self._photo: Optional[Any] = None
        self.image_path: Optional[Path] = None

        root.title("GUI neuro-symbolic — preview")
        root.minsize(520, 480)

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(main)
        row1.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row1, text="Choose image…", command=self._pick_image).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="")
        ttk.Label(row1, textvariable=self.path_var, wraplength=420).pack(
            side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True
        )

        ttk.Label(main, text="Instruction").pack(anchor=tk.W)
        self.task_var = tk.StringVar(value="")
        self.task_entry = ttk.Entry(main, textvariable=self.task_var)
        self.task_entry.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Run model", command=self._run).pack(side=tk.LEFT)
        ttk.Label(
            btn_row,
            text=f"Checkpoint: {checkpoint.name}",
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.status = tk.StringVar(value="Load an image, type a task, then Run.")
        ttk.Label(main, textvariable=self.status, foreground="#333").pack(anchor=tk.W)

        self.canvas = tk.Canvas(main, background="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _ensure_model(self) -> None:
        if self.model is not None and self.cfg is not None:
            return
        ck_path = self.checkpoint
        if not ck_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        cfg = ModelConfig()
        if isinstance(ck, dict) and "cfg" in ck and isinstance(ck["cfg"], dict):
            cfg = ModelConfig(**ck["cfg"])
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model, _ = build_model(cfg)
        model.load_state_dict(sd)
        model.to(self.device)
        model.eval()
        self.cfg = cfg
        self.model = model

    def _pick_image(self) -> None:
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self.image_path = Path(p)
        self.path_var.set(str(self.image_path))
        self.status.set("Image selected. Enter instruction and click Run.")

    @torch.no_grad()
    def _run(self) -> None:
        if self.image_path is None or not self.image_path.is_file():
            messagebox.showwarning("Missing image", "Choose an image file first.")
            return
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Missing instruction", "Enter an instruction.")
            return
        try:
            self._ensure_model()
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            return
        assert self.model is not None and self.cfg is not None
        cfg = self.cfg

        self.status.set("Running…")
        self.root.update_idletasks()

        try:
            img_t = load_image(str(self.image_path), cfg.image_size).unsqueeze(0).to(self.device)
            task_ids = (
                task_text_to_ids(task, cfg.task_max_len, cfg.task_vocab_size)
                .unsqueeze(0)
                .to(self.device)
            )
            logits = self.model(img_t, task_ids)
            sym = self.model.predict_symbol_json(logits, 0)
            raw = neural_logits_to_action_dict(
                logits["action_type_logits"],
                logits["click_logits"],
                logits["key_logits"],
                cfg.grid_h,
                cfg.grid_w,
                batch_idx=0,
            )
        except Exception as e:
            self.status.set("Error.")
            messagebox.showerror("Inference failed", str(e))
            return

        base = _tensor_to_pil(img_t[0])
        drawn = _draw_overlay(base, sym, raw)
        shown = _fit_display(drawn, max_side=960)

        self._photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        self.canvas.config(width=max(shown.width + 20, 400), height=max(shown.height + 20, 300))
        self.canvas.create_image(
            shown.width // 2 + 10,
            shown.height // 2 + 10,
            image=self._photo,
            anchor=tk.CENTER,
        )

        cap = _action_caption(raw)
        self.status.set(f"Done — {cap}")

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "out" / "gui_neurosymbolic_psai_full.pt"),
        help="Path to train.py checkpoint",
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ck = Path(args.checkpoint).expanduser()
    if not ck.is_absolute():
        ck = (_ROOT / ck).resolve()
    root = tk.Tk()
    app = InferGuiApp(root, checkpoint=ck, device=args.device)
    app.run()


if __name__ == "__main__":
    main()
