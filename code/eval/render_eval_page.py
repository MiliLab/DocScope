"""144 DPI page rendering + box-drawing helpers for score_bbox.py.

- Gold bboxes: absolute pixels at 144 DPI.
- Pred bboxes: normalized [0,1] (x1,y1,x2,y2).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "pdfs"
DEFAULT_CACHE = PROJECT_ROOT / "page_cache" / "144dpi"
DPI = 144


def _font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_pdf_page_144(doc_id: str, page: int,
                        cache_root: Path = DEFAULT_CACHE) -> Path | None:
    """Render a single 144 DPI PNG for `page` (1-based) of pdfs/<doc_id>.pdf."""
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return None
    cache = cache_root / doc_id
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"page_{page:04d}.png"
    if out_path.exists():
        return out_path
    prefix = cache / f"tmp_p{page:04d}"
    cmd = [
        "pdftoppm", "-r", str(DPI),
        "-f", str(page), "-l", str(page),
        "-png", str(pdf_path), str(prefix),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception:
        return None
    matches = sorted(cache.glob(f"{prefix.name}-*.png"))
    if not matches:
        return None
    try:
        matches[0].rename(out_path)
    except OSError:
        return None
    return out_path


def _draw_box(draw: ImageDraw.ImageDraw, bbox_px, color: str, label: str,
              width: int = 4) -> None:
    x1, y1, x2, y2 = bbox_px
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if not label:
        return
    font = _font(16)
    try:
        t = draw.textbbox((0, 0), label, font=font)
        tw, th = t[2] - t[0], t[3] - t[1]
    except Exception:
        tw, th = 20, 16
    pad = 2
    ty = max(0, y1 - th - 2 * pad)
    draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
    draw.text((x1 + pad, ty + pad), label, fill="white", font=font)


def render_pair(
    doc_id: str,
    page: int,
    gold_bboxes_px: List[List[float]],
    pred_bboxes_norm: List[List[float]],
    out_path: Path,
    cache_root: Path = DEFAULT_CACHE,
) -> bool:
    """Render page with green GOLD[i] boxes and red PRED boxes."""
    src = render_pdf_page_144(doc_id, page, cache_root=cache_root)
    if src is None:
        return False
    im = Image.open(src).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for i, gb in enumerate(gold_bboxes_px or []):
        if not gb or len(gb) != 4:
            continue
        label = "GOLD" if len(gold_bboxes_px) == 1 else f"GOLD[{i+1}]"
        _draw_box(draw, tuple(gb), "green", label)
    pred_count = len([b for b in pred_bboxes_norm if b and len(b) == 4])
    for idx, b in enumerate(pred_bboxes_norm):
        if not b or len(b) != 4:
            continue
        box_px = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
        label = "PRED" if pred_count == 1 else f"PRED[{idx+1}]"
        _draw_box(draw, box_px, "red", label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return True
