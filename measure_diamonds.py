#!/usr/bin/env python3
"""
Detect diamonds in a product render, draw contour circles with mm measurements,
and compare against CAD spec from the .3dm file.

Usage:
  python3 measure_diamonds.py DPBN0001/FRONT.jpg --px-per-mm 14.93
  python3 measure_diamonds.py DPBN0001/3D.jpg    --px-per-mm 14.93
"""
import sys, os, argparse, math
from PIL import Image, ImageDraw, ImageFont

# ── Confirmed specs from DPBN0001.3dm binary extraction ─────────────────────
SPECS = {
    "bangle_outer_diameter_mm": 60.5,
    "diamond_small_mm":          3.85,
    "diamond_large_mm":          4.85,
    "tolerance_pct":             0.10,
}


def find_diamond_blobs_cad(img_rgb, px_per_mm):
    """
    Detect diamonds in a CAD image by their cyan/teal colour
    (Matrix CAD uses J_Cyan material for diamonds).
    Returns list of (cx, cy, radius, pixel_count) sorted left→right.
    """
    w, h    = img_rgb.size
    pixels  = img_rgb.load()
    diamond_pixels = []

    for x in range(w):
        for y in range(h):
            r, g, b = pixels[x, y][:3]
            # Cyan: high G and B, low R
            if b > 100 and g > 100 and r < 150 and b > r + 30 and g > r + 20:
                diamond_pixels.append((x, y))

    if not diamond_pixels:
        return []

    # Cluster into blobs
    blobs = []
    used  = set()
    for sx, sy in diamond_pixels:
        if (sx, sy) in used:
            continue
        cluster = [(sx, sy)]
        used.add((sx, sy))
        queue = [(sx, sy)]
        while queue:
            cx, cy = queue.pop()
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]:
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in used: continue
                if 0 <= nx < w and 0 <= ny < h:
                    nr, ng, nb = pixels[nx, ny][:3]
                    if nb > 100 and ng > 100 and nr < 150 and nb > nr+20:
                        used.add((nx, ny))
                        cluster.append((nx, ny))
                        queue.append((nx, ny))

        if len(cluster) > 10:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            cx_c = sum(xs) // len(xs)
            cy_c = sum(ys) // len(ys)
            rx   = (max(xs) - min(xs)) / 2
            ry   = (max(ys) - min(ys)) / 2
            r    = (rx + ry) / 2
            blobs.append((cx_c, cy_c, r, len(cluster)))

    # Merge nearby
    merged = []
    used_b = [False] * len(blobs)
    for i, (x1,y1,r1,n1) in enumerate(blobs):
        if used_b[i]: continue
        group = [(x1,y1,r1,n1)]; used_b[i] = True
        for j, (x2,y2,r2,n2) in enumerate(blobs):
            if used_b[j]: continue
            if math.hypot(x1-x2, y1-y2) < 60:
                group.append((x2,y2,r2,n2)); used_b[j] = True
        total_n = sum(g[3] for g in group)
        mx = sum(g[0]*g[3] for g in group) // total_n
        my = sum(g[1]*g[3] for g in group) // total_n
        mr = max(g[2] for g in group)
        merged.append((mx, my, mr, total_n))

    return sorted(merged, key=lambda b: b[0])


def annotate(img_path, px_per_mm):
    img   = Image.open(img_path).convert("RGB")
    gray  = img.convert("L")
    w, h  = img.size

    # Detect diamonds — use CAD colour detection (cyan = diamond in Matrix CAD)
    diamonds = find_diamond_blobs_cad(img, px_per_mm)

    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_sm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_big = ImageFont.load_default()
        font_sm  = font_big

    spec_sizes = [SPECS["diamond_small_mm"], SPECS["diamond_large_mm"]]
    tol        = SPECS["tolerance_pct"]

    print(f"\n{'─'*60}")
    print(f"File: {img_path}   |   Scale: {px_per_mm} px/mm")
    print(f"CAD spec  — small diamond: {SPECS['diamond_small_mm']}mm  "
          f"large diamond: {SPECS['diamond_large_mm']}mm  "
          f"bangle: {SPECS['bangle_outer_diameter_mm']}mm")
    print(f"Tolerance: ±{tol*100:.0f}%")
    print(f"{'─'*60}")

    for idx, (cx, cy, r, n) in enumerate(diamonds):
        measured_mm = (r * 2) / px_per_mm
        spec_mm     = spec_sizes[idx] if idx < len(spec_sizes) else spec_sizes[-1]
        low         = spec_mm * (1 - tol)
        high        = spec_mm * (1 + tol)
        ok          = low <= measured_mm <= high
        verdict     = "PASS ✓" if ok else "FAIL ✗"
        color       = (0, 180, 0) if ok else (220, 30, 30)

        # Draw contour circle
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=3)

        # Draw crosshair
        draw.line([(cx-r-10, cy), (cx+r+10, cy)], fill=color, width=1)
        draw.line([(cx, cy-r-10), (cx, cy+r+10)], fill=color, width=1)

        # Dimension line across diameter
        draw.line([(cx-r, cy+r+15), (cx+r, cy+r+15)], fill=color, width=2)
        draw.line([(cx-r, cy+r+10), (cx-r, cy+r+20)], fill=color, width=2)
        draw.line([(cx+r, cy+r+10), (cx+r, cy+r+20)], fill=color, width=2)

        label = f"{measured_mm:.2f}mm  (spec {spec_mm}mm)  {verdict}"
        draw.text((cx - r, cy + r + 24), label, fill=color, font=font_sm)

        print(f"Diamond {idx+1}:  measured={measured_mm:.2f}mm  "
              f"spec={spec_mm}mm  range={low:.2f}–{high:.2f}mm  → {verdict}")

    # ── Scale bar ────────────────────────────────────────────────────────────
    bar_mm    = 10
    bar_px    = int(bar_mm * px_per_mm)
    bar_x     = 30
    bar_y     = h - 50
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + 8], fill=(30, 30, 30))
    draw.text((bar_x, bar_y + 12), f"{bar_mm}mm", fill=(30, 30, 30), font=font_sm)

    # ── Bangle width indicator (horizontal span) ─────────────────────────────
    mid_y = h // 2
    row   = [gray.getpixel((x, mid_y)) for x in range(w)]
    bng_l = next((x for x in range(w)   if row[x] < 220), None)
    bng_r = next((x for x in range(w-1, -1, -1) if row[x] < 220), None)
    if bng_l and bng_r:
        bng_mm = (bng_r - bng_l) / px_per_mm
        ok_b   = abs(bng_mm - SPECS["bangle_outer_diameter_mm"]) / SPECS["bangle_outer_diameter_mm"] <= tol
        col_b  = (0, 150, 0) if ok_b else (220, 30, 30)
        draw.line([(bng_l, mid_y - 15), (bng_r, mid_y - 15)], fill=col_b, width=2)
        draw.line([(bng_l, mid_y-20), (bng_l, mid_y-10)], fill=col_b, width=2)
        draw.line([(bng_r, mid_y-20), (bng_r, mid_y-10)], fill=col_b, width=2)
        draw.text((bng_l + (bng_r-bng_l)//2 - 40, mid_y - 38),
                  f"bangle {bng_mm:.1f}mm (spec {SPECS['bangle_outer_diameter_mm']}mm)",
                  fill=col_b, font=font_sm)
        v = "PASS ✓" if ok_b else "FAIL ✗"
        print(f"Bangle  :  measured={bng_mm:.1f}mm  spec={SPECS['bangle_outer_diameter_mm']}mm  → {v}")

    print(f"{'─'*60}\n")

    # Save
    name, ext = os.path.splitext(os.path.basename(img_path))
    out = f"measured_{name}.jpg"
    img.save(out)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images",       nargs="+")
    p.add_argument("--px-per-mm",  type=float, default=14.93)
    args = p.parse_args()
    for img in args.images:
        annotate(img, args.px_per_mm)
