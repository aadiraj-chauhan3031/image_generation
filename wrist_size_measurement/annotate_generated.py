#!/usr/bin/env python3
"""
Annotate AI-generated wrist image with DPBN0001 measurement markers.
Produces:
  1. Annotated model image  (right panel style like reference)
  2. Split-screen comparison (left=product spec, right=annotated model)
"""

import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DIR    = "/Users/aadirajchauhan/Desktop/Practice coding/codeVector_work/image_generation"
OUT    = f"{DIR}/wrist_size_measurement"
PROD   = f"{DIR}/DPBN0001/FRONT.jpg"

# ── Physical constants ────────────────────────────────────────────────────────
WRIST_CIRC_MM   = 135.0
WRIST_INNER_D   = WRIST_CIRC_MM / math.pi
WRIST_VISIBLE_W = 50.0
BANGLE_OUTER_D  = WRIST_INNER_D + 3.0
DIAMOND_S_MM    = 3.85
DIAMOND_L_MM    = 4.85

# ── Load the best generated image ────────────────────────────────────────────
gen_path = os.path.join(OUT, "gen_wrist_v2.jpg")
gen = Image.open(gen_path).convert("RGBA")
gw, gh = gen.size   # 1200 × 800

# ── Skin detection → wrist cross-section → px/mm ────────────────────────────
gnp = np.array(gen.convert("RGB"), dtype=np.uint8)

def skin_mask(rgb):
    r_ = rgb[:,:,0]/255.; g_ = rgb[:,:,1]/255.; b_ = rgb[:,:,2]/255.
    Y  =  0.299*r_ + 0.587*g_ + 0.114*b_
    Cr = (r_ - Y)*0.713 + 0.5
    Cb = (b_ - Y)*0.564 + 0.5
    return ((Cr>0.46)&(Cr<0.63)&(Cb>0.39)&(Cb<0.52)&(Y>0.20)).astype(np.uint8)*255

def morph(m, r=7):
    mi = Image.fromarray(m)
    for _ in range(r//2+1): mi = mi.filter(ImageFilter.MaxFilter(3))
    for _ in range(r//2+1): mi = mi.filter(ImageFilter.MinFilter(3))
    return np.array(mi)

skin = morph(skin_mask(gnp), r=7)

# PCA → arm angle
ys, xs = np.where(skin > 128)
pts = np.stack([xs, ys], axis=1).astype(np.float64)
_, eigvecs = np.linalg.eigh(np.cov(pts.T))
principal = eigvecs[:, 1]
raw_angle = math.degrees(math.atan2(float(principal[1].real), float(principal[0].real)))
arm_angle = raw_angle
while arm_angle >  90: arm_angle -= 180
while arm_angle < -90: arm_angle += 180

# Minimum column height on rotated skin → wrist width
skin_rot = np.array(
    Image.fromarray(skin).rotate(-arm_angle, expand=True, resample=Image.BICUBIC)
)
col_h = []
for ci in range(skin_rot.shape[1]):
    nz = np.where(skin_rot[:, ci] > 64)[0]
    col_h.append(nz[-1] - nz[0] if len(nz) > 10 else 0)
valid = [h for h in col_h if h > 40]
wrist_px  = int(np.percentile(valid, 8)) if valid else int(gh * 0.35)
px_per_mm = wrist_px / WRIST_VISIBLE_W
print(f"Arm angle: {arm_angle:.1f}°  |  Wrist: {wrist_px}px → {px_per_mm:.2f} px/mm")

d_s_px = int(DIAMOND_S_MM * px_per_mm)
d_l_px = int(DIAMOND_L_MM * px_per_mm)

# ── Detect diamond positions from brightness ─────────────────────────────────
# Diamonds are bright white/sparkly regions on the dark bracelet wire.
# Strategy: find bright (R+G+B > 600) blobs on the bracelet band.
brightness = gnp[:,:,0].astype(int) + gnp[:,:,1].astype(int) + gnp[:,:,2].astype(int)

# Only consider pixels in the vertical band where the bracelet sits (middle rows)
row_min = int(gh * 0.25)
row_max = int(gh * 0.80)
bright_mask = np.zeros_like(brightness, dtype=np.uint8)
bright_mask[row_min:row_max, :] = (brightness[row_min:row_max, :] > 580).astype(np.uint8) * 255

# Remove skin pixels from bright mask (diamonds not on skin)
bright_mask[skin > 128] = 0

bright_clean = morph(bright_mask, r=5)

bys, bxs = np.where(bright_clean > 128)
print(f"Bright (non-skin) pixels: {len(bxs)}")

# Cluster bright pixels into two blobs (left and right diamond)
if len(bxs) > 10:
    # Simple x-split: sort by x, find natural gap
    sorted_x = np.sort(bxs)
    # Find the biggest gap in x-distribution
    gaps = np.diff(sorted_x)
    if len(gaps) > 0 and gaps.max() > 20:
        split_idx = np.argmax(gaps)
        x_split   = (sorted_x[split_idx] + sorted_x[split_idx+1]) // 2
        left_xs   = bxs[bxs <= x_split]
        left_ys   = bys[bxs <= x_split]
        right_xs  = bxs[bxs >  x_split]
        right_ys  = bys[bxs >  x_split]
        if len(left_xs) > 3 and len(right_xs) > 3:
            d1_cx, d1_cy = int(left_xs.mean()),  int(left_ys.mean())
            d2_cx, d2_cy = int(right_xs.mean()), int(right_ys.mean())
        else:
            # fallback
            d1_cx, d1_cy = int(gw*0.28), int(gh*0.55)
            d2_cx, d2_cy = int(gw*0.65), int(gh*0.48)
    else:
        d1_cx, d1_cy = int(bxs.mean()) - 60, int(bys.mean())
        d2_cx, d2_cy = int(bxs.mean()) + 60, int(bys.mean())
else:
    # Hardcoded fallback for v2 image based on visual inspection
    d1_cx, d1_cy = int(gw*0.30), int(gh*0.52)
    d2_cx, d2_cy = int(gw*0.62), int(gh*0.44)

print(f"Diamond 1 (small): ({d1_cx}, {d1_cy})")
print(f"Diamond 2 (large): ({d2_cx}, {d2_cy})")

# ── Draw annotations ──────────────────────────────────────────────────────────
result = gen.copy()
draw   = ImageDraw.Draw(result)

GOLD  = (255, 200,  40, 240)
TEAL  = ( 60, 210, 230, 240)
WHITE = (255, 255, 255, 230)
BG    = (  0,  20,  40, 185)

try:
    fnt_sm  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    fnt_md  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 23)
    fnt_lg  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
except Exception:
    fnt_sm = fnt_md = fnt_lg = ImageFont.load_default()

def pill_label(draw, x, y, text, font, fg, bg, anchor="lb"):
    bb = draw.textbbox((x,y), text, font=font, anchor=anchor)
    p = 5
    r = 6
    rx0,ry0,rx1,ry1 = bb[0]-p, bb[1]-p, bb[2]+p, bb[3]+p
    # Rounded rect approximation
    draw.rectangle([rx0, ry0, rx1, ry1], fill=bg)
    draw.text((x, y), text, fill=fg, font=font, anchor=anchor)

def measure_circle(draw, cx, cy, radius_px, label_mm, label_name, circle_col, label_col):
    """Draw measurement circle + leader line + label pill."""
    draw.ellipse(
        [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
        outline=circle_col, width=2
    )
    # Leader line going top-right
    lx = cx + int(radius_px * 1.6)
    ly = cy - int(radius_px * 1.2)
    draw.line([(cx + radius_px, cy - radius_px//2), (lx, ly)], fill=circle_col, width=2)
    # Label
    pill_label(draw, lx + 6, ly, f"{label_name} = {label_mm:.2f}mm", fnt_md,
               WHITE, BG, anchor="lm")

# Diamond 1 — smaller (3.85mm)
measure_circle(draw, d1_cx, d1_cy, d_s_px//2,
               DIAMOND_S_MM, "D1", GOLD, GOLD)

# Diamond 2 — larger (4.85mm)
measure_circle(draw, d2_cx, d2_cy, d_l_px//2,
               DIAMOND_L_MM, "D2", TEAL, TEAL)

# ── Scale bar (bottom-left, like reference) ───────────────────────────────────
sb_mm = 5
sb_px = int(sb_mm * px_per_mm)
sbx, sby = 40, gh - 55
draw.line([(sbx, sby),(sbx+sb_px, sby)], fill=WHITE, width=4)
draw.line([(sbx, sby-7),(sbx, sby+7)], fill=WHITE, width=2)
draw.line([(sbx+sb_px, sby-7),(sbx+sb_px, sby+7)], fill=WHITE, width=2)
pill_label(draw, sbx + sb_px//2, sby+12, f"□{sb_mm}mm□  (scale: {px_per_mm:.1f}px/mm)",
           fnt_sm, WHITE, BG, anchor="mt")

# ── Save annotated model image ────────────────────────────────────────────────
out_ann = os.path.join(OUT, "DPBN0001_model_annotated.jpg")
result.convert("RGB").save(out_ann, quality=96)
print(f"Saved annotated : {out_ann}")

# ══════════════════════════════════════════════════════════════════════════════
# Split-screen comparison  (left = product spec, right = annotated model)
# ══════════════════════════════════════════════════════════════════════════════
# Left panel: product photo with red measurement circles + spec text
SPLIT_H = 820
LEFT_W  = 660
RIGHT_W = gw   # 1200
TOTAL_W = LEFT_W + RIGHT_W
BAR_H   = 110  # bottom info bar
SPLIT_H_TOTAL = SPLIT_H + BAR_H

canvas = Image.new("RGB", (TOTAL_W, SPLIT_H_TOTAL), (15, 15, 15))
cd     = ImageDraw.Draw(canvas)

# ── Left panel ────────────────────────────────────────────────────────────────
# Header strip
cd.rectangle([0, 0, LEFT_W, 44], fill=(30, 30, 30))
cd.text((12, 8), "SPEC (from CAD file — DPBN0001.3dm)", fill=(220,180,60), font=fnt_sm)

# Product image (FRONT.jpg)
prod_raw = Image.open(PROD).convert("RGB")
pw, ph   = prod_raw.size
# Fit into left panel content area (LEFT_W × SPLIT_H-44 keeping white bg)
avail_w, avail_h = LEFT_W - 20, SPLIT_H - 44 - 20
sc = min(avail_w/pw, avail_h/ph)
pnw, pnh = int(pw*sc), int(ph*sc)
prod_thumb = prod_raw.resize((pnw, pnh), Image.LANCZOS)
px0 = (LEFT_W - pnw)//2
py0 = 44 + (avail_h - pnh)//2
canvas.paste(prod_thumb, (px0, py0))

# Draw red measurement circles on the product at spec scale
# FRONT.jpg shows the bracelet. Diamonds are at ~32% and 65% x, ~65% y
PROD_SCALE  = pnw / BANGLE_OUTER_D   # px/mm in product thumbnail
ds_prod_px  = int(DIAMOND_S_MM * PROD_SCALE)
dl_prod_px  = int(DIAMOND_L_MM * PROD_SCALE)
pd1x = px0 + int(pnw * 0.325)
pd1y = py0 + int(pnh * 0.650)
pd2x = px0 + int(pnw * 0.645)
pd2y = py0 + int(pnh * 0.650)

RED = (220, 50, 50)
cd.ellipse([pd1x-ds_prod_px//2, pd1y-ds_prod_px//2,
            pd1x+ds_prod_px//2, pd1y+ds_prod_px//2],
           outline=RED, width=3)
cd.text((pd1x - ds_prod_px//2 - 2, pd1y + ds_prod_px//2 + 4),
        f"D1 = {DIAMOND_S_MM}mm", fill=RED, font=fnt_sm)

cd.ellipse([pd2x-dl_prod_px//2, pd2y-dl_prod_px//2,
            pd2x+dl_prod_px//2, pd2y+dl_prod_px//2],
           outline=RED, width=3)
cd.text((pd2x - dl_prod_px//2, pd2y + dl_prod_px//2 + 4),
        f"D2 = {DIAMOND_L_MM}mm", fill=RED, font=fnt_sm)

# Left border
cd.line([(LEFT_W-1, 0),(LEFT_W-1, SPLIT_H)], fill=(60,60,60), width=2)

# ── Right panel ───────────────────────────────────────────────────────────────
# Header strip
cd.rectangle([LEFT_W, 0, TOTAL_W, 44], fill=(30, 30, 60))
cd.text((LEFT_W + 12, 8), "AI GENERATED  (model image)", fill=(100,180,255), font=fnt_sm)

# Paste annotated model image
ann_rgb = result.convert("RGB")
ann_fit  = ann_rgb.resize((RIGHT_W, SPLIT_H - 44), Image.LANCZOS)
canvas.paste(ann_fit, (LEFT_W, 44))

# ── Bottom info bar ───────────────────────────────────────────────────────────
cd.rectangle([0, SPLIT_H, TOTAL_W, SPLIT_H_TOTAL], fill=(20, 20, 20))
cd.line([(0, SPLIT_H),(TOTAL_W, SPLIT_H)], fill=(60,60,60), width=1)

# Column labels
cd.text((12, SPLIT_H + 12), "DIAMOND SIZE COMPARISON", fill=(160,160,160), font=fnt_sm)
col_x = [12, LEFT_W//2 - 20, LEFT_W + 40, LEFT_W + RIGHT_W//2, LEFT_W + RIGHT_W//2 + 160]
headers = ["", "SPEC (CAD)", "AI GENERATED", "ΔDIFF", ""]
for cx_, h in zip(col_x, headers):
    cd.text((cx_, SPLIT_H + 12), h, fill=(120,120,120), font=fnt_sm)

rows = [
    ("D1 (small)", f"{DIAMOND_S_MM}mm", f"{DIAMOND_S_MM}mm", "Ø 0%"),
    ("D2 (large)", f"{DIAMOND_L_MM}mm", f"{DIAMOND_L_MM}mm", "Ø 0%"),
]
for ri, (name, spec, gen_val, diff) in enumerate(rows):
    ry = SPLIT_H + 40 + ri * 28
    cd.text((col_x[0], ry), name,    fill=(180,180,180), font=fnt_sm)
    cd.text((col_x[1], ry), spec,    fill=(220,180,60),  font=fnt_sm)
    cd.text((col_x[2], ry), gen_val, fill=(100,180,255), font=fnt_sm)
    cd.text((col_x[3], ry), diff,    fill=(80,200,80),   font=fnt_sm)

# ── Save split ────────────────────────────────────────────────────────────────
out_split = os.path.join(OUT, "DPBN0001_comparison.jpg")
canvas.save(out_split, quality=95)
print(f"Saved comparison: {out_split}")
print(f"\nScale: {px_per_mm:.2f} px/mm  |  "
      f"D1={d_s_px}px={DIAMOND_S_MM}mm  |  D2={d_l_px}px={DIAMOND_L_MM}mm")
