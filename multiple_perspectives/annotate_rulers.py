#!/usr/bin/env python3
"""
Draws mm rulers next to diamonds in both product and lifestyle halves
of the DPBN0002 and DPBN0003 split comparison images.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np, os

OUT = os.path.join(os.path.dirname(__file__), "output")
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"

PROD_PX_MM   = 14.95   # from bangle outer diameter (912px ≈ 61mm)
LIFE_PX_MM_3 = 4.86    # dpbn0003 lifestyle, from diamond-gap reference
LIFE_PX_MM_2 = 5.09    # dpbn0002 lifestyle, from diamond-gap reference

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def outline_text(draw, xy, text, font, fill, outline_color, stroke=2):
    x, y = xy
    for dx in range(-stroke, stroke+1):
        for dy in range(-stroke, stroke+1):
            if dx or dy:
                draw.text((x+dx, y+dy), text, font=font, fill=outline_color)
    draw.text(xy, text, font=font, fill=fill)


def ruler_v(draw, cx, cy, h_px, label, color, font, offset_x=18, side='right'):
    """Vertical bracket ruler (height) beside a diamond."""
    tick = 10
    x = cx + offset_x if side == 'right' else cx - offset_x
    y0, y1 = int(cy - h_px/2), int(cy + h_px/2)
    draw.line([(x, y0), (x, y1)], fill=color, width=2)
    draw.line([(x-tick, y0), (x+tick, y0)], fill=color, width=2)
    draw.line([(x-tick, y1), (x+tick, y1)], fill=color, width=2)
    bb = font.getbbox(label)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = x + tick + 4 if side == 'right' else x - tick - 4 - tw
    outline_text(draw, (tx, int(cy - th/2)), label, font, color, (0,0,0))


def ruler_h(draw, cx, cy, w_px, label, color, font, offset_y=18, side='below'):
    """Horizontal bracket ruler (width) below/above a diamond."""
    tick = 10
    y = cy + offset_y if side == 'below' else cy - offset_y
    x0, x1 = int(cx - w_px/2), int(cx + w_px/2)
    draw.line([(x0, y), (x1, y)], fill=color, width=2)
    draw.line([(x0, y-tick), (x0, y+tick)], fill=color, width=2)
    draw.line([(x1, y-tick), (x1, y+tick)], fill=color, width=2)
    bb = font.getbbox(label)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    outline_text(draw, (int(cx - tw/2), y + tick + 2), label, font, color, (0,0,0))


# ---------------------------------------------------------------------------
# DPBN0003 — twin baguettes 3.0 × 6.0 mm
# ---------------------------------------------------------------------------

def annotate_dpbn0003():
    img = Image.open(os.path.join(OUT, "split_dpbn0003.jpg")).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_lg = ImageFont.truetype(FONT_PATH, 26)
    font_sm = ImageFont.truetype(FONT_PATH, 20)

    # ---- Product side (LEFT panel, x=0-1199, 1:1 scale) ----
    # Detected clusters: left center=(369,725) right center=(871,725)
    PROD_COLOR = (220, 20, 60)   # crimson — visible on white

    for cx, cy, cluster_x1 in [(369, 725, 480), (871, 725, 987)]:
        w_px = round(3.0 * PROD_PX_MM)   # 45 px = 3.0 mm
        h_px = round(6.0 * PROD_PX_MM)   # 90 px = 6.0 mm
        # Vertical ruler: right of cluster bbox
        ruler_v(draw, cluster_x1, cy, h_px, "6.0 mm", PROD_COLOR, font_lg,
                offset_x=20, side='right')
        # Horizontal ruler: below cluster
        ruler_h(draw, cx, 767, w_px, "3.0 mm", PROD_COLOR, font_lg,
                offset_y=16, side='below')

    # ---- Lifestyle side (RIGHT panel, x=1220-2185) ----
    # Original lifestyle (928×1152) scaled to 966×1200 → offset_x=1220
    sx = 966 / 928     # x scale factor
    sy = 1200 / 1152   # y scale factor
    OX = 1220          # x offset in split image
    LIFE_COLOR = (255, 220, 0)  # golden yellow — visible on skin

    # Cluster 0: original center=(535,631), bbox x1=572, bbox y1=664
    # Cluster 1: original center=(680,556), bbox x1=699, bbox y1=580
    for (orig_cx, orig_cy, orig_bx1, orig_by1) in [
        (535, 631, 572, 664),
        (680, 556, 699, 580),
    ]:
        cx = int(orig_cx * sx) + OX
        cy = int(orig_cy * sy)
        bx1 = int(orig_bx1 * sx) + OX
        by1 = int(orig_by1 * sy)
        w_px = round(3.0 * LIFE_PX_MM_3)
        h_px = round(6.0 * LIFE_PX_MM_3)
        ruler_v(draw, bx1, cy, h_px, "6.0 mm", LIFE_COLOR, font_sm,
                offset_x=16, side='right')
        ruler_h(draw, cx, by1, w_px, "3.0 mm", LIFE_COLOR, font_sm,
                offset_y=12, side='below')

    out_path = os.path.join(OUT, "annotated_dpbn0003.jpg")
    img.save(out_path, quality=95)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# DPBN0002 — pear 4.0×5.9 mm + emerald 3.2×4.4 mm
# ---------------------------------------------------------------------------

def annotate_dpbn0002():
    img = Image.open(os.path.join(OUT, "split_dpbn0002.jpg")).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_lg = ImageFont.truetype(FONT_PATH, 26)
    font_sm = ImageFont.truetype(FONT_PATH, 20)

    # ---- Lifestyle side (LEFT panel, x=0-965) ----
    sx = 966 / 928
    sy = 1200 / 1152
    LIFE_COLOR = (255, 220, 0)

    # Cluster 0 (larger = pear): center=(524,638), bbox x1=554, by1=669
    # Cluster 1 (smaller = emerald): center=(673,554), bbox x1=693, by1=576
    for (orig_cx, orig_cy, orig_bx1, orig_by1, wm, hm, lbl) in [
        (524, 638, 554, 669,  4.0, 5.9, ("4.0", "5.9")),   # pear
        (673, 554, 693, 576,  3.2, 4.4, ("3.2", "4.4")),   # emerald
    ]:
        cx  = int(orig_cx * sx)
        cy  = int(orig_cy * sy)
        bx1 = int(orig_bx1 * sx)
        by1 = int(orig_by1 * sy)
        w_px = round(wm * LIFE_PX_MM_2)
        h_px = round(hm * LIFE_PX_MM_2)
        ruler_v(draw, bx1, cy, h_px, f"{hm} mm", LIFE_COLOR, font_sm,
                offset_x=14, side='right')
        ruler_h(draw, cx, by1, w_px, f"{wm} mm", LIFE_COLOR, font_sm,
                offset_y=10, side='below')

    # ---- Product side (RIGHT panel, x=986-2185, 1:1 scale) ----
    OX = 986
    PROD_COLOR = (220, 20, 60)

    # Detected clusters: left (pear) center=(344,776) bx1=466 by1=814
    #                    right (emerald) center=(905,776) bx1=1030 by1=814
    for (cx, cy, bx1, by1, wm, hm) in [
        (344,  776, 466,  814, 4.0, 5.9),   # pear (left)
        (905,  776, 1030, 814, 3.2, 4.4),   # emerald (right)
    ]:
        split_cx  = cx  + OX
        split_bx1 = bx1 + OX
        split_by1 = by1
        w_px = round(wm * PROD_PX_MM)
        h_px = round(hm * PROD_PX_MM)
        ruler_v(draw, split_bx1, cy, h_px, f"{hm} mm", PROD_COLOR, font_lg,
                offset_x=20, side='right')
        ruler_h(draw, split_cx, split_by1, w_px, f"{wm} mm", PROD_COLOR, font_lg,
                offset_y=16, side='below')

    out_path = os.path.join(OUT, "annotated_dpbn0002.jpg")
    img.save(out_path, quality=95)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    annotate_dpbn0003()
    annotate_dpbn0002()
    print("Done.")
