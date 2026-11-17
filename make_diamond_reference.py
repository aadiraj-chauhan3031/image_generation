#!/usr/bin/env python3
"""
Build DPBN0001_diamond_reference.jpg — an annotated FRONT.jpg with:
  • Red circles around each diamond at the correct spec diameters
  • Dimension lines showing 3.85mm and 4.85mm
  • 10mm scale bar
  • Title banner
"""
import sys, math
from PIL import Image, ImageDraw, ImageFont

FRONT      = "DPBN0001/FRONT.jpg"
OUT        = "DPBN0001_diamond_reference.jpg"
PX_PER_MM  = 14.93

SPEC = {
    "d1_mm": 3.85,   # left diamond
    "d2_mm": 4.85,   # right diamond
    "bangle_mm": 60.5,
}

# Approximate seed centres from visual inspection of FRONT.jpg (1200×1200)
SEED_L = (430, 215)
SEED_R = (665, 215)
SEARCH_RADIUS = 60   # px — local search window


def find_stone_centre(pixels, seed, w, h, search=60):
    """Return centroid of dark-on-white pixels near seed (the metal prongs + stone facets)."""
    sx, sy = seed
    pts = []
    for x in range(max(0, sx - search), min(w, sx + search)):
        for y in range(max(0, sy - search), min(h, sy + search)):
            r, g, b = pixels[x, y][:3]
            # Non-white: either the prong metal (grey) or stone (slight sparkle)
            brightness = (r + g + b) / 3
            if brightness < 230:
                pts.append((x, y))
    if not pts:
        return seed
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    return (cx, cy)


def arrow_line(draw, x1, y1, x2, y2, color, width=2, head=8):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # arrowhead at end
    angle = math.atan2(y2 - y1, x2 - x1)
    for da in (0.4, -0.4):
        ax = x2 - head * math.cos(angle - da)
        ay = y2 - head * math.sin(angle - da)
        draw.line([(x2, y2), (int(ax), int(ay))], fill=color, width=width)


def main():
    img    = Image.open(FRONT).convert("RGB")
    pixels = img.load()
    w, h   = img.size
    draw   = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_sm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_xs  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:
        font_big = font_sm = font_xs = ImageFont.load_default()

    RED   = (220, 20, 20)
    BLACK = (20, 20, 20)

    # ── Find stone centres ───────────────────────────────────────────────────
    c1 = find_stone_centre(pixels, SEED_L, w, h, SEARCH_RADIUS)
    c2 = find_stone_centre(pixels, SEED_R, w, h, SEARCH_RADIUS)

    r1 = (SPEC["d1_mm"] * PX_PER_MM) / 2   # spec radius in px
    r2 = (SPEC["d2_mm"] * PX_PER_MM) / 2

    print(f"D1 centre: {c1}  radius: {r1:.1f}px = {SPEC['d1_mm']}mm")
    print(f"D2 centre: {c2}  radius: {r2:.1f}px = {SPEC['d2_mm']}mm")

    # ── Draw circles ────────────────────────────────────────────────────────
    for (cx, cy), r in ((c1, r1), (c2, r2)):
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=RED, width=3
        )

    # ── Dimension line for D1 (left diamond) ────────────────────────────────
    # Horizontal dimension line above the circle
    cx1, cy1 = c1
    dim_y1 = int(cy1 - r1 - 30)
    # tick lines
    draw.line([(int(cx1 - r1), dim_y1 - 8), (int(cx1 - r1), dim_y1 + 8)], fill=RED, width=2)
    draw.line([(int(cx1 + r1), dim_y1 - 8), (int(cx1 + r1), dim_y1 + 8)], fill=RED, width=2)
    # dimension line with arrows
    arrow_line(draw, int(cx1 - r1), dim_y1, int(cx1 + r1), dim_y1, RED)
    label1 = f"D1 = {SPEC['d1_mm']}mm"
    bbox = draw.textbbox((0, 0), label1, font=font_sm)
    lw = bbox[2] - bbox[0]
    draw.text((cx1 - lw // 2, dim_y1 - 24), label1, fill=RED, font=font_sm)

    # ── Dimension line for D2 (right diamond) ───────────────────────────────
    cx2, cy2 = c2
    dim_y2 = int(cy2 - r2 - 30)
    draw.line([(int(cx2 - r2), dim_y2 - 8), (int(cx2 - r2), dim_y2 + 8)], fill=RED, width=2)
    draw.line([(int(cx2 + r2), dim_y2 - 8), (int(cx2 + r2), dim_y2 + 8)], fill=RED, width=2)
    arrow_line(draw, int(cx2 - r2), dim_y2, int(cx2 + r2), dim_y2, RED)
    label2 = f"D2 = {SPEC['d2_mm']}mm"
    bbox = draw.textbbox((0, 0), label2, font=font_sm)
    lw = bbox[2] - bbox[0]
    draw.text((cx2 - lw // 2, dim_y2 - 24), label2, fill=RED, font=font_sm)

    # ── Leader lines from circle edge to labels ──────────────────────────────
    draw.line([(int(cx1), int(cy1 + r1 + 3)), (int(cx1), int(cy1 + r1 + 20))], fill=RED, width=1)
    draw.text((int(cx1) + 5, int(cy1 + r1 + 20)), "D1", fill=RED, font=font_xs)
    draw.line([(int(cx2), int(cy2 + r2 + 3)), (int(cx2), int(cy2 + r2 + 20))], fill=RED, width=1)
    draw.text((int(cx2) + 5, int(cy2 + r2 + 20)), "D2", fill=RED, font=font_xs)

    # ── 10mm scale bar (bottom-left) ─────────────────────────────────────────
    bar_mm = 10
    bar_px = int(bar_mm * PX_PER_MM)
    bx, by = 40, h - 70
    # alternating black/white segments
    seg = bar_px // 5
    for i in range(5):
        fill = BLACK if i % 2 == 0 else (255, 255, 255)
        draw.rectangle([bx + i * seg, by, bx + (i + 1) * seg, by + 12], fill=fill, outline=BLACK)
    draw.rectangle([bx, by, bx + bar_px, by + 12], outline=BLACK, width=2)
    draw.text((bx, by + 16), f"◀ {bar_mm}mm ▶", fill=BLACK, font=font_xs)

    # ── Title banner ─────────────────────────────────────────────────────────
    banner_h = 40
    draw.rectangle([(0, 0), (w, banner_h)], fill=(240, 240, 240))
    draw.line([(0, banner_h), (w, banner_h)], fill=RED, width=2)
    title = (f"DPBN0001  |  D1 = {SPEC['d1_mm']}mm  |  D2 = {SPEC['d2_mm']}mm  |"
             f"  Bangle OD = {SPEC['bangle_mm']}mm  |  MAINTAIN EXACT SIZES")
    draw.text((w // 2 - 350, 8), title, fill=BLACK, font=font_sm)

    img.save(OUT, quality=95)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
