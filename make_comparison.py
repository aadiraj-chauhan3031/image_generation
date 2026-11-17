#!/usr/bin/env python3
"""
Build a side-by-side comparison image:
  Left  — Image #7: TOP.jpg product render, annotated with CAD spec sizes
  Right — Image #6: AI-generated model image, annotated with measured sizes
  Bottom strip — comparison table
"""
import math
from PIL import Image, ImageDraw, ImageFont

IMG7 = "DPBN0001/TOP.jpg"          # product spec
IMG6 = "output/output_222617_2.jpeg"  # AI generated model
OUT  = "output/comparison_diamond_sizes.jpg"

# CAD-confirmed specs
PX_PER_MM_PRODUCT = 14.93   # TOP.jpg render scale
PX_PER_MM_MODEL   = 4.25    # model photos scale

SPEC_D1_MM = 3.85
SPEC_D2_MM = 4.85

# ── Visually-identified diamond centres in TOP.jpg (1200×1200) ───────────────
# Top/front view — gap at bottom, diamonds at ~y=760
TOP_D1 = (395, 760)   # left  diamond (3.85mm)
TOP_D2 = (650, 762)   # right diamond (4.85mm)

# ── Visually-identified diamond centres in model image (1213×864) ────────────
# Located by scanning for cool-bright (non-skin) pixels
MODEL_D1 = (694, 492)   # lower diamond cluster
MODEL_D2 = (878, 425)   # upper diamond cluster


def find_local_diamond_centre(pixels, seed, w, h, search=45, warm=False):
    """Return centroid of bright/sparkle pixels near seed (diamonds are bright vs skin)."""
    sx, sy = seed
    pts = []
    for x in range(max(0, sx - search), min(w, sx + search)):
        for y in range(max(0, sy - search), min(h, sy + search)):
            r, g, b = pixels[x, y][:3]
            if warm:
                # Cool-bright vs warm skin: key signal is high blue channel
                # skin has r-b > 70; metal/diamond has r-b < 60 and b > 150
                if b > 150 and r > 150 and g > 140 and (r - b) < 60:
                    pts.append((x, y))
            else:
                if r > 200 and g > 200 and b > 200:
                    pts.append((x, y))
    if len(pts) < 5:
        return seed
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    return (cx, cy)


def measure_stone_radius(pixels, centre, w, h, px_per_mm, search=50, warm=False):
    """Estimate radius as half the span of bright pixels around centre."""
    cx, cy = centre
    bright = []
    for x in range(max(0, cx - search), min(w, cx + search)):
        for y in range(max(0, cy - search), min(h, cy + search)):
            r, g, b = pixels[x, y][:3]
            if warm:
                if b > 150 and r > 150 and g > 140 and (r - b) < 60:
                    bright.append((x, y))
            else:
                if r > 190 and g > 190 and b > 190:
                    bright.append((x, y))
    if len(bright) < 4:
        return None
    xs = [p[0] for p in bright]
    ys = [p[1] for p in bright]
    rx = (max(xs) - min(xs)) / 2
    ry = (max(ys) - min(ys)) / 2
    r_px = (rx + ry) / 2
    return r_px, r_px * 2 / px_per_mm   # (radius_px, diameter_mm)


def arrow_dim_line(draw, cx, cy, r, color, label, font, above=True):
    """Draw a horizontal dimension line above/below the circle with label."""
    offset = int(r) + 25 if above else int(r) + 25
    y = int(cy - offset) if above else int(cy + offset)
    x1, x2 = int(cx - r), int(cx + r)
    # end ticks
    draw.line([(x1, y - 7), (x1, y + 7)], fill=color, width=2)
    draw.line([(x2, y - 7), (x2, y + 7)], fill=color, width=2)
    # dimension arrow line
    draw.line([(x1, y), (x2, y)], fill=color, width=2)
    # arrowheads
    for sx, ex in [(x1, x1 + 10), (x2, x2 - 10)]:
        draw.line([(sx, y), (ex, y - 5)], fill=color, width=2)
        draw.line([(sx, y), (ex, y + 5)], fill=color, width=2)
    # label centred
    bbox = draw.textbbox((0, 0), label, font=font)
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]
    draw.rectangle([cx - lw//2 - 3, y - lh - 4, cx + lw//2 + 3, y - 1],
                   fill=(0, 0, 0, 180) if hasattr(color, '__len__') else (0, 0, 0))
    draw.text((cx - lw//2, y - lh - 2), label, fill=color, font=font)


def annotate_product(img7_path):
    img = Image.open(img7_path).convert("RGB")
    pixels = img.load()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font_lbl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_hdr = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
        font_sm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_lbl = font_hdr = font_sm = ImageFont.load_default()

    RED = (220, 30, 30)

    # Refine centres
    c1 = find_local_diamond_centre(pixels, TOP_D1, w, h, 40)
    c2 = find_local_diamond_centre(pixels, TOP_D2, w, h, 40)

    r1_px = (SPEC_D1_MM * PX_PER_MM_PRODUCT) / 2
    r2_px = (SPEC_D2_MM * PX_PER_MM_PRODUCT) / 2

    # Circles
    for (cx, cy), r in ((c1, r1_px), (c2, r2_px)):
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=RED, width=3)

    # Dimension lines
    arrow_dim_line(draw, c1[0], c1[1], r1_px, RED, f"D1 = {SPEC_D1_MM}mm", font_lbl, above=False)
    arrow_dim_line(draw, c2[0], c2[1], r2_px, RED, f"D2 = {SPEC_D2_MM}mm", font_lbl, above=False)

    # Header banner
    draw.rectangle([(0, 0), (w, 48)], fill=(20, 20, 20))
    draw.text((20, 10), "SPEC (from CAD file — DPBN0001.3dm)", fill=(255, 220, 60), font=font_hdr)

    return img, (c1, r1_px, SPEC_D1_MM), (c2, r2_px, SPEC_D2_MM)


def annotate_model(img6_path):
    img = Image.open(img6_path).convert("RGB")
    pixels = img.load()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font_lbl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_hdr = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
        font_sm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_lbl = font_hdr = font_sm = ImageFont.load_default()

    BLUE = (30, 140, 255)

    # Calibrate scale from inter-diamond pixel distance
    # CAD gap between diamond centres = 17.08mm
    CAD_GAP_MM = 17.08
    gap_px = math.hypot(MODEL_D2[0] - MODEL_D1[0], MODEL_D2[1] - MODEL_D1[1])
    px_per_mm = gap_px / CAD_GAP_MM  # ~11.06 px/mm

    c1 = find_local_diamond_centre(pixels, MODEL_D1, w, h, 40, warm=True)
    c2 = find_local_diamond_centre(pixels, MODEL_D2, w, h, 40, warm=True)

    # Measure span of cool-bright pixels in 25px window
    def span_mm(centre):
        cx, cy = centre
        pts = []
        for x in range(max(0, cx-40), min(w, cx+40)):
            for y in range(max(0, cy-40), min(h, cy+40)):
                r, g, b = pixels[x, y][:3]
                # cool-bright (diamond/metal) OR any near-white sparkle pixel
                if (b > 130 and r > 130 and g > 120 and (r - b) < 70) or (r > 210 and g > 210 and b > 210):
                    pts.append((x, y))
        if not pts:
            return None, None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        avg_r_px = ((max(xs)-min(xs))/2 + (max(ys)-min(ys))/2) / 2
        return avg_r_px, avg_r_px * 2 / px_per_mm

    r1_px, d1_mm = span_mm(c1)
    r2_px, d2_mm = span_mm(c2)

    r1_px = r1_px or (SPEC_D1_MM * px_per_mm / 2)
    r2_px = r2_px or (SPEC_D2_MM * px_per_mm / 2)

    for (cx, cy), r in ((c1, r1_px), (c2, r2_px)):
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=BLUE, width=3)

    lbl1 = f"D1 ≈ {d1_mm:.2f}mm" if d1_mm else "D1 = ?"
    lbl2 = f"D2 ≈ {d2_mm:.2f}mm" if d2_mm else "D2 = ?"
    arrow_dim_line(draw, c1[0], c1[1], r1_px, BLUE, lbl1, font_lbl, above=True)
    arrow_dim_line(draw, c2[0], c2[1], r2_px, BLUE, lbl2, font_lbl, above=True)

    # Scale bar using calibrated scale
    bar_mm = 5
    bar_px = int(bar_mm * px_per_mm)
    bx, by = 20, h - 55
    draw.rectangle([bx, by, bx+bar_px, by+8], fill=(255,255,255))
    draw.text((bx, by+12), f"◀ {bar_mm}mm ▶  (scale: {px_per_mm:.1f}px/mm)", fill=(255,255,255), font=font_sm)

    draw.rectangle([(0, 0), (w, 48)], fill=(20, 20, 20))
    draw.text((20, 10), "AI GENERATED (model image)", fill=(100, 200, 255), font=font_hdr)

    return img, (c1, r1_px, d1_mm), (c2, r2_px, d2_mm)


def make_comparison_strip(w, spec1, spec2, gen1, gen2):
    """Bottom strip comparing spec vs generated sizes."""
    h = 110
    strip = Image.new("RGB", (w, h), (15, 15, 15))
    draw = ImageDraw.Draw(strip)
    try:
        font_b = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    YELLOW = (255, 220, 60)
    CYAN   = (100, 200, 255)
    WHITE  = (230, 230, 230)
    RED    = (220, 80, 80)
    GREEN  = (80, 220, 80)

    draw.text((20, 10), "DIAMOND SIZE COMPARISON", fill=WHITE, font=font_b)
    draw.line([(0, 38), (w, 38)], fill=(60, 60, 60), width=1)

    col_w = w // 4

    headers = ["", "SPEC (CAD)", "AI GENERATED", "ΔDIFF"]
    x_positions = [20, col_w, col_w * 2, col_w * 3]
    for i, hdr in enumerate(headers):
        clr = YELLOW if i == 1 else (CYAN if i == 2 else WHITE)
        draw.text((x_positions[i], 42), hdr, fill=clr, font=font_s)

    # D1 row
    draw.text((20, 68), "D1 (small)", fill=WHITE, font=font_s)
    draw.text((x_positions[1], 68), f"{spec1:.2f}mm", fill=YELLOW, font=font_s)
    if gen1:
        draw.text((x_positions[2], 68), f"{gen1:.2f}mm", fill=CYAN, font=font_s)
        diff1 = abs(gen1 - spec1) / spec1 * 100
        col = GREEN if diff1 <= 10 else RED
        draw.text((x_positions[3], 68), f"{'✓' if diff1<=10 else '✗'} {diff1:.0f}%", fill=col, font=font_s)
    else:
        draw.text((x_positions[2], 68), "—", fill=CYAN, font=font_s)

    # D2 row
    draw.text((20, 88), "D2 (large)", fill=WHITE, font=font_s)
    draw.text((x_positions[1], 88), f"{spec2:.2f}mm", fill=YELLOW, font=font_s)
    if gen2:
        draw.text((x_positions[2], 88), f"{gen2:.2f}mm", fill=CYAN, font=font_s)
        diff2 = abs(gen2 - spec2) / spec2 * 100
        col = GREEN if diff2 <= 10 else RED
        draw.text((x_positions[3], 88), f"{'✓' if diff2<=10 else '✗'} {diff2:.0f}%", fill=col, font=font_s)
    else:
        draw.text((x_positions[2], 88), "—", fill=CYAN, font=font_s)

    return strip


def main():
    img7_ann, s1, s2 = annotate_product(IMG7)
    img6_ann, g1, g2 = annotate_model(IMG6)

    print(f"Product  D1: {s1[2]:.2f}mm   D2: {s2[2]:.2f}mm")
    print(f"AI gen   D1: {g1[2]:.2f}mm   D2: {g2[2]:.2f}mm" if g1[2] and g2[2]
          else "AI gen: could not measure")

    # Scale both to same height
    target_h = max(img7_ann.height, img6_ann.height)
    def fit_h(img, h):
        r = h / img.height
        return img.resize((int(img.width * r), h), Image.LANCZOS)

    left  = fit_h(img7_ann, target_h)
    right = fit_h(img6_ann, target_h)

    divider = 6
    total_w = left.width + divider + right.width
    strip_h = 110

    canvas = Image.new("RGB", (total_w, target_h + strip_h), (0, 0, 0))
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width + divider, 0))

    strip = make_comparison_strip(total_w, s1[2], s2[2], g1[2], g2[2])
    canvas.paste(strip, (0, target_h))

    canvas.save(OUT, quality=95)
    print(f"\nSaved: {OUT}  ({canvas.size})")


if __name__ == "__main__":
    main()
