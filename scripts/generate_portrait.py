"""
Photo -> animated ASCII portrait SVG.

Usage:
    python generate_portrait.py photo.jpg -o ascii.svg

Pipeline (matches the guide's spec):
  1. rembg cut-out              -> background forced to pure white
  2. bilateral filter           -> smooths skin, keeps edges
  3. CLAHE, clip=3.0            -> local contrast per tile
  4. darkening curve (v/255)^1.7 -> keeps glasses/brows/lips from washing out
  5. map brightness -> 13-char ramp, leading space clears background to nothing
  6. render as a self-typing animated SVG (SMIL clipPath wipe, staggered rows)
"""

import argparse
import io

import cv2
import numpy as np
from PIL import Image
from rembg import remove

# 13-level ramp, lightest (background) to darkest. Index 0 must stay a space
# so the removed background renders as nothing, not a visible glyph.
RAMP = [" ", ".", "`", "'", '"', ",", ":", ";", "+", "*", "#", "%", "@"]

COLS = 90
CHAR_ASPECT = 0.48  # rows = cols * (h/w) * CHAR_ASPECT, monospace chars ~2x tall as wide
FONT_SIZE = 12.9
CHAR_W = 0.600 * FONT_SIZE  # advance width baked in at 0.600em (JetBrains/Liberation/DejaVu/Noto)
ROW_H = FONT_SIZE * 1.05
STAGGER = 0.09  # seconds between each row's typing start
TYPE_DURATION = 0.6  # seconds for a single row to wipe in


def cutout_to_white(img_bytes: bytes) -> Image.Image:
    """Remove background, composite the subject onto pure white."""
    fg = remove(img_bytes)
    fg_img = Image.open(io.BytesIO(fg)).convert("RGBA")
    white_bg = Image.new("RGBA", fg_img.size, (255, 255, 255, 255))
    white_bg.paste(fg_img, (0, 0), fg_img)
    return white_bg.convert("RGB")


def process_grayscale(rgb_img: Image.Image) -> np.ndarray:
    """bilateral filter -> CLAHE -> darkening curve. Returns uint8 grayscale array."""
    gray = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2GRAY)

    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(smoothed)

    curved = (np.power(contrasted.astype(np.float64) / 255.0, 1.7) * 255.0)
    return curved.astype(np.uint8)


def to_ascii_rows(gray: np.ndarray, cols: int = COLS) -> list[str]:
    h, w = gray.shape
    rows = max(1, round(cols * (h / w) * CHAR_ASPECT))

    resized = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    ramp_len = len(RAMP)
    ascii_rows = []
    for row in resized:
        # bright pixel -> low ramp index (space / background), dark pixel -> high index
        line = "".join(RAMP[min(ramp_len - 1, int(v / 256 * ramp_len))] for v in (255 - row))
        ascii_rows.append(line)
    return ascii_rows


def rows_to_svg(rows: list[str]) -> str:
    cols = max(len(r) for r in rows)
    width = cols * CHAR_W
    height = len(rows) * ROW_H

    row_elems = []
    for i, text in enumerate(rows):
        y = (i + 1) * ROW_H - (ROW_H * 0.25)
        clip_id = f"clip{i}"
        begin = round(i * STAGGER, 3)
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        row_elems.append(f"""
  <clipPath id="{clip_id}">
    <rect x="0" y="{y - ROW_H:.2f}" height="{ROW_H:.2f}" width="0">
      <animate attributeName="width" from="0" to="{width:.2f}"
               begin="{begin}s" dur="{TYPE_DURATION}s" fill="freeze" />
    </rect>
  </clipPath>
  <text class="a" x="0" y="{y:.2f}" clip-path="url(#{clip_id})"
        font-family="monospace" font-size="{FONT_SIZE}"
        xml:space="preserve">{escaped}</text>""")

    # GitHub renders SVGs as standalone <img> - currentColor resolves to black
    # with no page context, so light/dark must be hardcoded via media query.
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}"
     width="{width:.2f}" height="{height:.2f}">
  <style>.a{{fill:#6e7681}}@media(prefers-color-scheme:dark){{.a{{fill:#c9d1d9}}}}</style>
{''.join(row_elems)}
</svg>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to your input photo (1200px+ recommended)")
    parser.add_argument("-o", "--output", default="ascii.svg")
    parser.add_argument("--cols", type=int, default=COLS)
    args = parser.parse_args()

    with open(args.photo, "rb") as f:
        photo_bytes = f.read()

    print("Removing background...")
    white_composited = cutout_to_white(photo_bytes)

    print("Applying bilateral filter + CLAHE + darkening curve...")
    gray = process_grayscale(white_composited)

    print(f"Mapping to {args.cols}-column ASCII ramp...")
    rows = to_ascii_rows(gray, cols=args.cols)

    print("Building animated SVG...")
    svg = rows_to_svg(rows)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Done -> {args.output} ({len(rows)} rows x {args.cols} cols)")


if __name__ == "__main__":
    main()
