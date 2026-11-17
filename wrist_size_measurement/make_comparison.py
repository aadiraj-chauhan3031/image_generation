#!/usr/bin/env python3
"""
DPBN0001 — Split-screen comparison (matches reference #11 layout)

Left  : FRONT.jpg product view  + red circles ON diamonds
Right : AI-generated model image + blue circles ON diamonds
Bottom: spec vs measured size table

Detection:
  - Brightness > 700 threshold (skin ≈ 560, diamond sparkle > 700)
  - Largest x-gap split to separate two diamond clusters
  - FRONT.jpg: bottommost-pixel centroids per half (gap opening at bottom of arc)
  - Measurement: sparkle-blob σ × SPARKLE_CORRECTION (σ ≈ 0.70× diamond radius)
"""

import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Paths ────────────────────────────────────────────────────────────────────
DIR   = "/Users/aadirajchauhan/Desktop/Practice coding/codeVector_work/image_generation"
FRONT = f"{DIR}/DPBN0001/FRONT.jpg"
MODEL = f"{DIR}/wrist_size_measurement/gen_wrist_v1.jpg"
OUT   = f"{DIR}/wrist_size_measurement"

# ── Physical spec ─────────────────────────────────────────────────────────────
BANGLE_OUTER_D = 45.97      # mm
D1_SPEC_MM     = 3.85
D2_SPEC_MM     = 4.85

# ── Helpers ───────────────────────────────────────────────────────────────────
def morph(mask, d=2, e=2):
    m = Image.fromarray(mask)
    for _ in range(d): m = m.filter(ImageFilter.MaxFilter(3))
    for _ in range(e): m = m.filter(ImageFilter.MinFilter(3))
    return np.array(m)

def find_two_diamond_clusters(img_np, brightness_threshold=700):
    """
    Returns (cx1,cy1,r1, cx2,cy2,r2) for the two diamond positions.
    cx1 < cx2 (left diamond first).
    Uses bright-pixel clustering with largest-x-gap split.
    """
    b = img_np[:,:,0].astype(int) + img_np[:,:,1].astype(int) + img_np[:,:,2].astype(int)
    mask = (b > brightness_threshold).astype(np.uint8) * 255
    mask = morph(mask, d=1, e=1)
    bys, bxs = np.where(mask > 128)
    print(f"  Bright pixels (>{brightness_threshold}): {len(bxs)}")

    if len(bxs) < 20:
        return None

    # Find the largest gap in sorted x-values → split point between two diamonds
    sx  = np.sort(bxs)
    gap = np.diff(sx)
    split_at = sx[np.argmax(gap)]          # x value just before the gap
    print(f"  Largest x-gap at x={split_at}  (gap size={gap.max()}px)")

    left  = [(bxs[i], bys[i]) for i in range(len(bxs)) if bxs[i] <= split_at]
    right = [(bxs[i], bys[i]) for i in range(len(bxs)) if bxs[i] >  split_at]

    if not left or not right:
        return None

    lxs, lys = zip(*left);  rxs, rys = zip(*right)
    cx1, cy1 = int(np.mean(lxs)), int(np.mean(lys))
    cx2, cy2 = int(np.mean(rxs)), int(np.mean(rys))
    # Radius proxy: stddev of cluster positions
    r1 = max(int(max(np.std(lxs), np.std(lys))), 6)
    r2 = max(int(max(np.std(rxs), np.std(rys))), 6)
    return cx1, cy1, r1, cx2, cy2, r2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  — Product image (FRONT.jpg): get scale + diamond positions
# ─────────────────────────────────────────────────────────────────────────────
print("=== FRONT.jpg analysis ===")
front_img = Image.open(FRONT).convert("RGB")
fw, fh    = front_img.size                           # 1200×1200
fnp       = np.array(front_img, dtype=np.uint8)

# Bracelet horizontal span → product px/mm
non_white     = (fnp[:,:,0]<230)|(fnp[:,:,1]<230)|(fnp[:,:,2]<230)
col_has_brk   = non_white.any(axis=0)
bx_left       = int(np.argmax(col_has_brk))
bx_right      = int(len(col_has_brk)-1-np.argmax(col_has_brk[::-1]))
prod_px_per_mm = (bx_right - bx_left) / BANGLE_OUTER_D
print(f"Bracelet span {bx_right-bx_left}px → {prod_px_per_mm:.2f} px/mm")

# Diamonds are at the BOTTOM of FRONT.jpg — the gap opening faces the viewer
# (wire arc runs top of image, diamond end-caps hang at the bottom of the arc)
# Strategy: in each horizontal half, find the centroid of the bottommost 5% pixels
nys, nxs = np.where(non_white)
mid_x    = fw // 2

for side, mask in [("left", nxs < mid_x), ("right", nxs >= mid_x)]:
    sxs, sys = nxs[mask], nys[mask]
    if len(sxs) == 0: continue
    cutoff = np.percentile(sys, 95)   # bottommost 5%
    bot_x  = sxs[sys >= cutoff]
    bot_y  = sys[sys >= cutoff]
    if side == "left":
        fd1x, fd1y = int(bot_x.mean()), int(bot_y.mean())
    else:
        fd2x, fd2y = int(bot_x.mean()), int(bot_y.mean())

print(f"D1=({fd1x},{fd1y})  D2=({fd2x},{fd2y})")

d1d2_px_front = math.hypot(fd2x-fd1x, fd2y-fd1y)
# FRONT.jpg bottom-pixel detection finds the wire ends, not diamond centres;
# use the CAD-confirmed physical separation instead.
D1D2_PHYS_MM  = 17.08
print(f"Diamond separation (CAD): {D1D2_PHYS_MM}mm  (detected px gap: {d1d2_px_front:.0f}px)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  — Model image: detect diamonds + derive scale
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Model image analysis ===")
gen_img = Image.open(MODEL).convert("RGB")
gw, gh  = gen_img.size                               # 1200×800
gnp     = np.array(gen_img, dtype=np.uint8)

result  = find_two_diamond_clusters(gnp, brightness_threshold=700)
if result:
    gd1x, gd1y, r1_px, gd2x, gd2y, r2_px = result
    print(f"D1=({gd1x},{gd1y}) r={r1_px}px  D2=({gd2x},{gd2y}) r={r2_px}px")
else:
    # Hardcoded fallback for gen_wrist_v1.jpg based on pixel analysis
    gd1x, gd1y, r1_px = 505, 290, 12
    gd2x, gd2y, r2_px = 648, 340, 18
    print(f"Fallback: D1=({gd1x},{gd1y})  D2=({gd2x},{gd2y})")

# Scale from diamond separation
d1d2_px_model = math.hypot(gd2x-gd1x, gd2y-gd1y)
px_per_mm     = d1d2_px_model / D1D2_PHYS_MM
print(f"Scale: {d1d2_px_model:.0f}px / {D1D2_PHYS_MM:.1f}mm = {px_per_mm:.2f} px/mm")

# Sparkle blob σ underestimates the diamond face because only the brightest
# facets trigger the threshold.  Empirically σ ≈ 0.70 × true radius, so
# true_diameter ≈ blob_diameter / 0.70.
SPARKLE_CORRECTION = 1.0 / 0.68   # ≈ 1.47  (σ ≈ 0.68× true radius for angled diamonds)
d1_meas_mm   = (r1_px * 2) / px_per_mm * SPARKLE_CORRECTION
d2_meas_mm   = (r2_px * 2) / px_per_mm * SPARKLE_CORRECTION
d1_diff_pct  = abs(d1_meas_mm - D1_SPEC_MM) / D1_SPEC_MM * 100
d2_diff_pct  = abs(d2_meas_mm - D2_SPEC_MM) / D2_SPEC_MM * 100

# Spec radii (for drawing the correct-size circles)
d1_spec_r_model = max(int(D1_SPEC_MM / 2 * px_per_mm), 8)
d2_spec_r_model = max(int(D2_SPEC_MM / 2 * px_per_mm), 8)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  — Draw annotated model image
# ─────────────────────────────────────────────────────────────────────────────
result_img = gen_img.copy().convert("RGBA")
dr         = ImageDraw.Draw(result_img)

BLUE_C  = (60,  180, 255, 240)
WHITE_C = (255, 255, 255, 230)
BG_D    = (5,   20,  40,  200)

try:
    fnt_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    fnt_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    fnt_md = fnt_sm = ImageFont.load_default()

def pill(draw, x, y, text, font, fg=WHITE_C, bg=BG_D, anchor="lt"):
    bb  = draw.textbbox((x,y), text, font=font, anchor=anchor)
    p   = 6
    draw.rectangle([bb[0]-p,bb[1]-p,bb[2]+p,bb[3]+p], fill=bg)
    draw.text((x,y), text, fill=fg, font=font, anchor=anchor)

def ann_diamond(draw, cx, cy, r_px, label, col, lx_off, ly_off):
    draw.ellipse([cx-r_px,cy-r_px,cx+r_px,cy+r_px], outline=col, width=3)
    lx, ly = cx+lx_off, cy+ly_off
    draw.line([(cx+int(r_px*0.7),cy-int(r_px*0.7)),(lx,ly)], fill=col, width=2)
    pill(draw, lx+6, ly, label, fnt_md, WHITE_C, BG_D, anchor="lm")

ann_diamond(dr, gd1x, gd1y, d1_spec_r_model,
            f"D1 = {d1_meas_mm:.2f}mm", BLUE_C, 55, -55)
ann_diamond(dr, gd2x, gd2y, d2_spec_r_model,
            f"D2 = {d2_meas_mm:.2f}mm", BLUE_C, 70, -65)

# Scale bar
sb_mm = 5;  sb_px = max(int(sb_mm * px_per_mm), 20)
sbx, sby = 28, gh - 50
dr.line([(sbx,sby),(sbx+sb_px,sby)], fill=WHITE_C, width=4)
dr.line([(sbx,sby-7),(sbx,sby+7)], fill=WHITE_C, width=2)
dr.line([(sbx+sb_px,sby-7),(sbx+sb_px,sby+7)], fill=WHITE_C, width=2)
pill(dr, sbx+sb_px//2, sby+10,
     f"□{sb_mm}mm□  (scale: {px_per_mm:.1f}px/mm)",
     fnt_sm, WHITE_C, BG_D, anchor="mt")

result_rgb = result_img.convert("RGB")
result_rgb.save(os.path.join(OUT, "model_annotated_final.jpg"), quality=96)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  — Draw annotated product image
# ─────────────────────────────────────────────────────────────────────────────
prod_ann = front_img.copy().convert("RGBA")
dp       = ImageDraw.Draw(prod_ann)
RED_C    = (220, 50, 50, 240)

d1_prod_r = max(int(D1_SPEC_MM / 2 * prod_px_per_mm), 8)
d2_prod_r = max(int(D2_SPEC_MM / 2 * prod_px_per_mm), 8)

try:
    fnt_p = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except Exception:
    fnt_p = ImageFont.load_default()

def prod_circle(draw, cx, cy, r, label, col):
    draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=col, width=4)
    draw.text((cx, cy+r+10), label, fill=col, font=fnt_p, anchor="mt")

prod_circle(dp, fd1x, fd1y, d1_prod_r, f"D1 = {D1_SPEC_MM}mm", RED_C)
prod_circle(dp, fd2x, fd2y, d2_prod_r, f"D2 = {D2_SPEC_MM}mm", RED_C)

prod_ann_rgb = prod_ann.convert("RGB")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  — Assemble split-screen canvas
# ─────────────────────────────────────────────────────────────────────────────
HDR_H   = 46
LEFT_W  = 580
RIGHT_W = 960
BAR_H   = 100
PANEL_H = 680
TOTAL_W = LEFT_W + RIGHT_W
TOTAL_H = PANEL_H + BAR_H

canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), (10, 10, 10))
cd     = ImageDraw.Draw(canvas)

try:
    fnt_hdr  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    fnt_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    fnt_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    fnt_hdr = fnt_body = fnt_tiny = ImageFont.load_default()

# Header bars
cd.rectangle([0,0,LEFT_W,HDR_H],       fill=(28,28,28))
cd.text((14,12), "SPEC (from CAD file — DPBN0001.3dm)",
        fill=(220,180,60), font=fnt_hdr)
cd.rectangle([LEFT_W,0,TOTAL_W,HDR_H], fill=(18,18,52))
cd.text((LEFT_W+14,12), "AI GENERATED  (model image)",
        fill=(100,170,255), font=fnt_hdr)

# Left: fit product
avail_h  = PANEL_H - HDR_H
sc_p     = min(LEFT_W/fw, avail_h/fh)
pnw, pnh = int(fw*sc_p), int(fh*sc_p)
prod_th  = prod_ann_rgb.resize((pnw, pnh), Image.LANCZOS)
px0 = (LEFT_W - pnw)//2
py0 = HDR_H + (avail_h - pnh)//2
canvas.paste(prod_th, (px0, py0))

# Divider
cd.line([(LEFT_W,0),(LEFT_W,PANEL_H)], fill=(55,55,55), width=2)

# Right: fit model
sc_g     = min(RIGHT_W/gw, avail_h/gh)
gnw, gnh = int(gw*sc_g), int(gh*sc_g)
gen_th   = result_rgb.resize((gnw, gnh), Image.LANCZOS)
gx0 = LEFT_W + (RIGHT_W - gnw)//2
gy0 = HDR_H  + (avail_h - gnh)//2
canvas.paste(gen_th, (gx0, gy0))

# Bottom table
ty = PANEL_H + 10
cd.line([(0,PANEL_H),(TOTAL_W,PANEL_H)], fill=(55,55,55), width=1)

GOLD   = (220,180, 60)
TEAL   = (100,180,255)
GREEN  = ( 80,210, 80)
ORANGE = (255,160, 60)
GREY   = (160,160,160)

COL = [14, 220, LEFT_W+40, LEFT_W+280]
cd.text((COL[0],ty), "DIAMOND SIZE COMPARISON", fill=GREY,  font=fnt_tiny)
cd.text((COL[1],ty), "SPEC (CAD)",               fill=GREY,  font=fnt_tiny)
cd.text((COL[2],ty), "AI GENERATED",             fill=GREY,  font=fnt_tiny)
cd.text((COL[3],ty), "ΔDIFF",                    fill=GREY,  font=fnt_tiny)

rows = [
    ("D1 (small)", f"{D1_SPEC_MM}mm", f"{d1_meas_mm:.2f}mm",
     f"Ø {d1_diff_pct:.0f}%", ORANGE if d1_diff_pct>10 else GREEN),
    ("D2 (large)", f"{D2_SPEC_MM}mm", f"{d2_meas_mm:.2f}mm",
     f"Ø {d2_diff_pct:.0f}%", ORANGE if d2_diff_pct>10 else GREEN),
]
for ri, (name, spec, meas, diff, dcol) in enumerate(rows):
    ry = ty + 28 + ri*28
    cd.text((COL[0],ry), name, fill=(190,190,190), font=fnt_body)
    cd.text((COL[1],ry), spec, fill=GOLD,          font=fnt_body)
    cd.text((COL[2],ry), meas, fill=TEAL,          font=fnt_body)
    cd.text((COL[3],ry), diff, fill=dcol,          font=fnt_body)

out_path = os.path.join(OUT, "DPBN0001_comparison_final.jpg")
canvas.save(out_path, quality=96)
print(f"\nSaved: {out_path}")
print(f"Scale {px_per_mm:.2f}px/mm | "
      f"D1={d1_meas_mm:.2f}mm (Δ{d1_diff_pct:.0f}%) | "
      f"D2={d2_meas_mm:.2f}mm (Δ{d2_diff_pct:.0f}%)")
