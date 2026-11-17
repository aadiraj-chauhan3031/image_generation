#!/usr/bin/env python3
"""
DPBN0001 — Proportional Wrist Overlay + Measurement Diagram
=============================================================
Physical data from CAD (DPBN0001 (3.8-4.8 RD).jpg):
  Diamond 1 : 3.85 × 3.85 mm  (0.21 ct)
  Diamond 2 : 4.85 × 4.85 mm  (0.42 ct)
  Total gems : 0.63 ct  |  18 KT white gold: 5.55 g

Wrist assumption: female circumference 130-140 mm → target 135 mm
  → dorsal visible width ≈ 50 mm
  → inner bangle diameter = circumference/π ≈ 42.97 mm
"""

import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Paths ────────────────────────────────────────────────────────────────────
DIR = "/Users/aadirajchauhan/Desktop/Practice coding/codeVector_work/image_generation"
MODEL_PATH   = f"{DIR}/scale_meaurement/model/close-up_shot.png"
PRODUCT_PATH = f"{DIR}/DPBN0001/TOP.jpg"
OUT_DIR      = f"{DIR}/wrist_size_measurement"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Physical measurements (mm) ───────────────────────────────────────────────
WRIST_CIRC_MM   = 135.0
WRIST_INNER_D   = WRIST_CIRC_MM / math.pi          # 42.97 mm
WRIST_VISIBLE_W = 50.0   # dorsal visible width for 135 mm circumference
BANGLE_OUTER_D  = WRIST_INNER_D + 3.0              # ~46 mm (wire adds ~1.5 mm per side)
DIAMOND_S_MM    = 3.85
DIAMOND_L_MM    = 4.85

# ── Load images ───────────────────────────────────────────────────────────────
model_img   = Image.open(MODEL_PATH).convert("RGBA")
product_raw = Image.open(PRODUCT_PATH).convert("RGBA")
mw, mh = model_img.size
pw, ph = product_raw.size
print(f"Model   : {mw}×{mh}  |  Product: {pw}×{ph}")

# ── Remove white background from bracelet ────────────────────────────────────
arr = np.array(product_raw, dtype=np.uint8)
white_mask = (arr[:,:,0] > 230) & (arr[:,:,1] > 230) & (arr[:,:,2] > 230)
arr[white_mask, 3] = 0
alpha_img  = Image.fromarray(arr[:,:,3], 'L').filter(ImageFilter.GaussianBlur(1.2))
arr[:,:,3] = np.array(alpha_img)
product_clean = Image.fromarray(arr, 'RGBA')
bbox = product_clean.getbbox()
product_cropped = product_clean.crop(bbox)
pcw, pch = product_cropped.size

# ── Skin mask ─────────────────────────────────────────────────────────────────
model_np = np.array(model_img.convert("RGB"), dtype=np.uint8)

def skin_mask(rgb):
    r_ = rgb[:,:,0]/255.0; g_ = rgb[:,:,1]/255.0; b_ = rgb[:,:,2]/255.0
    Y  =  0.299*r_ + 0.587*g_ + 0.114*b_
    Cr = (r_ - Y)*0.713 + 0.5
    Cb = (b_ - Y)*0.564 + 0.5
    return ((Cr>0.49)&(Cr<0.62)&(Cb>0.40)&(Cb<0.52)&(Y>0.25)).astype(np.uint8)*255

def morph(mask, r=9):
    m = Image.fromarray(mask)
    for _ in range(r//2+1): m = m.filter(ImageFilter.MaxFilter(3))
    for _ in range(r//2+1): m = m.filter(ImageFilter.MinFilter(3))
    return np.array(m)

skin = morph(skin_mask(model_np), r=9)

# ── PCA on skin pixels → arm principal axis ────────────────────────────────
ys, xs = np.where(skin > 128)
pts = np.stack([xs, ys], axis=1).astype(np.float64)
mean_pt = pts.mean(axis=0)
_, eigvecs = np.linalg.eigh(np.cov(pts.T))
principal = eigvecs[:, 1]   # largest eigenvalue is last in eigh
raw_angle = math.degrees(math.atan2(float(principal[1].real),
                                     float(principal[0].real)))
# Normalise to (-90, 90] so the angle describes wrist tilt not direction
arm_angle = raw_angle
while arm_angle >  90: arm_angle -= 180
while arm_angle < -90: arm_angle += 180
print(f"Arm tilt: {arm_angle:.1f}°  (raw {raw_angle:.1f}°)")

# ── Rotate skin → arm horizontal → find minimum column height = wrist width ─
skin_rot = np.array(
    Image.fromarray(skin).rotate(-arm_angle, expand=True, resample=Image.BICUBIC)
)
col_h = []
for ci in range(skin_rot.shape[1]):
    nz = np.where(skin_rot[:, ci] > 64)[0]
    col_h.append(nz[-1] - nz[0] if len(nz) > 10 else 0)
valid = [h for h in col_h if h > 50]
wrist_px = int(np.percentile(valid, 8)) if valid else int(mh*0.20)
px_per_mm = wrist_px / WRIST_VISIBLE_W
print(f"Wrist cross-section: {wrist_px}px → {px_per_mm:.2f} px/mm")

# ── Locate existing bracelet (green stones) for placement ────────────────────
r_, g_, b_ = (model_np[:,:,i].astype(float) for i in range(3))
green = (g_ > 100) & (g_ > r_*1.1) & (g_ > b_*0.9) & (b_ > 60)
gys, gxs = np.where(green)
if len(gxs) > 20:
    brace_cx, brace_cy = int(gxs.mean()), int(gys.mean())
    print(f"Existing bracelet at ({brace_cx}, {brace_cy})")
else:
    brace_cx, brace_cy = int(mean_pt[0]), int(mean_pt[1])
    print(f"Bracelet fallback ({brace_cx}, {brace_cy})")

# ── Resize bracelet to true physical scale ───────────────────────────────────
bangle_target_px = int(BANGLE_OUTER_D * px_per_mm)
bracelet = product_cropped.resize(
    (bangle_target_px, int(bangle_target_px * pch / pcw)), Image.LANCZOS
)
bangle_w, bangle_h = bracelet.size
print(f"Bracelet: {bangle_w}×{bangle_h}px  ({BANGLE_OUTER_D:.1f}mm × {BANGLE_OUTER_D*pch/pcw:.1f}mm)")

# ── Rotate bracelet to match wrist tilt ──────────────────────────────────────
if abs(arm_angle) > 1.5:
    bracelet = bracelet.rotate(-arm_angle, expand=True, resample=Image.BICUBIC)
    bangle_w, bangle_h = bracelet.size

# ── Paste position ────────────────────────────────────────────────────────────
paste_x = max(0, min(brace_cx - bangle_w//2, mw - bangle_w))
paste_y = max(0, min(brace_cy - bangle_h//2, mh - bangle_h))

# ── Composite ─────────────────────────────────────────────────────────────────
result = model_img.copy()
result.paste(bracelet, (paste_x, paste_y), bracelet)

# ── Drawing helpers ───────────────────────────────────────────────────────────
draw = ImageDraw.Draw(result)
YELLOW = (255, 220, 50,  235)
CYAN   = (80,  220, 255, 235)
WHITE  = (255, 255, 255, 230)
BG     = (0,   0,   0,  180)
TICK   = 6

try:
    fnt_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    fnt_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
except Exception:
    fnt_sm = fnt_md = ImageFont.load_default()

def lbl(draw, xy, text, font, fg=WHITE, bg=BG):
    bb = draw.textbbox(xy, text, font=font, anchor="mm")
    p = 3
    draw.rectangle([bb[0]-p, bb[1]-p, bb[2]+p, bb[3]+p], fill=bg)
    draw.text(xy, text, fill=fg, font=font, anchor="mm")

def ruler_h(draw, x1, x2, y, label, color, above=True):
    draw.line([(x1,y),(x2,y)], fill=color, width=2)
    draw.line([(x1,y-TICK),(x1,y+TICK)], fill=color, width=2)
    draw.line([(x2,y-TICK),(x2,y+TICK)], fill=color, width=2)
    ly = y - TICK - 11 if above else y + TICK + 11
    lbl(draw, ((x1+x2)//2, ly), label, fnt_sm, WHITE, BG)

def ruler_v(draw, x, y1, y2, label, color, right=True):
    draw.line([(x,y1),(x,y2)], fill=color, width=2)
    draw.line([(x-TICK,y1),(x+TICK,y1)], fill=color, width=2)
    draw.line([(x-TICK,y2),(x+TICK,y2)], fill=color, width=2)
    lx = x + TICK + 42 if right else x - TICK - 42
    lbl(draw, (lx, (y1+y2)//2), label, fnt_sm, WHITE, BG)

# ── Diamond ruler positions ────────────────────────────────────────────────────
# Diamonds sit at ~27% and ~73% across the bracelet, ~80% down
d_s_px = int(DIAMOND_S_MM * px_per_mm)
d_l_px = int(DIAMOND_L_MM * px_per_mm)
ds_cx  = paste_x + int(bangle_w * 0.27)
dl_cx  = paste_x + int(bangle_w * 0.73)
gem_mid_y = paste_y + int(bangle_h * 0.78)

rb_y = paste_y + bangle_h + 18   # ruler baseline y

# Small diamond — rulers below the bracelet and to the left side
ruler_h(draw, ds_cx - d_s_px//2, ds_cx + d_s_px//2, rb_y,
        f"Ø {DIAMOND_S_MM} mm", YELLOW)
ruler_v(draw, ds_cx - d_s_px//2 - 18,
        gem_mid_y - d_s_px//2, gem_mid_y + d_s_px//2,
        f"Ø {DIAMOND_S_MM} mm", YELLOW, right=False)

# Large diamond — rulers below the bracelet and to the right side
ruler_h(draw, dl_cx - d_l_px//2, dl_cx + d_l_px//2, rb_y,
        f"Ø {DIAMOND_L_MM} mm", CYAN)
ruler_v(draw, dl_cx + d_l_px//2 + 18,
        gem_mid_y - d_l_px//2, gem_mid_y + d_l_px//2,
        f"Ø {DIAMOND_L_MM} mm", CYAN, right=True)

# ── Scale bar (bottom-right) ──────────────────────────────────────────────────
sb_mm = 10
sb_px = int(sb_mm * px_per_mm)
sbx, sby = mw - sb_px - 20, mh - 36
draw.line([(sbx,sby),(sbx+sb_px,sby)], fill=WHITE, width=3)
draw.line([(sbx,sby-6),(sbx,sby+6)], fill=WHITE, width=2)
draw.line([(sbx+sb_px,sby-6),(sbx+sb_px,sby+6)], fill=WHITE, width=2)
lbl(draw, (sbx+sb_px//2, sby-12), f"← {sb_mm} mm →", fnt_sm)

# ── Info panel (top-left) ─────────────────────────────────────────────────────
info = [
    ("DPBN0001 — Open Cuff Bangle",         WHITE),
    (f"Wrist circumference : {WRIST_CIRC_MM:.0f} mm",     WHITE),
    (f"Wrist inner Ø       : {WRIST_INNER_D:.1f} mm",     WHITE),
    (f"Diamond (small)  Ø : {DIAMOND_S_MM} mm  (0.21 ct)", YELLOW),
    (f"Diamond (large)  Ø : {DIAMOND_L_MM} mm  (0.42 ct)", CYAN),
    (f"18 KT white gold    : 5.55 g",        WHITE),
    (f"Scale               : {px_per_mm:.1f} px/mm",       (180,180,180,220)),
]
lh = 23
pw2 = 348; ph2 = len(info)*lh + 16
panel = Image.new("RGBA", (pw2, ph2), (0,0,0,170))
result.paste(panel, (12,12), panel)
for i,(txt,col) in enumerate(info):
    draw.text((20, 18 + i*lh), txt, fill=col, font=fnt_sm)

# ── Save composite ────────────────────────────────────────────────────────────
out1 = os.path.join(OUT_DIR, "DPBN0001_on_wrist.png")
result.convert("RGB").save(out1, quality=96)
print(f"Saved composite  : {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
# Measurement diagram — clean white background, proportional rulers
# ═══════════════════════════════════════════════════════════════════════════════
DIAG_W, DIAG_H = 1100, 780
diag = Image.new("RGB", (DIAG_W, DIAG_H), (250, 250, 250))
dd   = ImageDraw.Draw(diag)

try:
    fnt_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    fnt_body  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    fnt_tiny  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    fnt_bold  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except Exception:
    fnt_title = fnt_body = fnt_tiny = fnt_bold = ImageFont.load_default()

# Title + subtitle
dd.text((DIAG_W//2, 22), "DPBN0001 — Measurement Diagram", fill=(25,25,25),
        font=fnt_title, anchor="mt")
dd.text((DIAG_W//2, 58), "Open Cuff Bangle  ·  18 KT White Gold  ·  2 Round Diamonds",
        fill=(110,110,110), font=fnt_tiny, anchor="mt")

# Use the ORIGINAL un-rotated bracelet thumbnail for a cleaner look
product_diag_raw = Image.open(PRODUCT_PATH).convert("RGBA")
arr_d = np.array(product_diag_raw, dtype=np.uint8)
wm_d  = (arr_d[:,:,0]>230)&(arr_d[:,:,1]>230)&(arr_d[:,:,2]>230)
arr_d[wm_d,3] = 0
product_diag = Image.fromarray(arr_d,'RGBA')
pb = product_diag.getbbox()
product_diag = product_diag.crop(pb)

# Fit thumbnail into a 580 × 300 box, maintaining aspect ratio
TH_MAX_W, TH_MAX_H = 580, 300
pd_w, pd_h = product_diag.size
sc = min(TH_MAX_W/pd_w, TH_MAX_H/pd_h)
th_w, th_h = int(pd_w*sc), int(pd_h*sc)
thumb_diag = product_diag.resize((th_w, th_h), Image.LANCZOS)

# Paste onto diagram (centered horizontally)
TX = (DIAG_W - th_w) // 2
TY = 85
bg_th = Image.new("RGBA", (th_w, th_h), (250,250,250,255))
bg_th.paste(thumb_diag, (0,0), thumb_diag)
diag.paste(bg_th.convert("RGB"), (TX, TY))

# Scale in diagram image (px/mm)
DS = th_w / BANGLE_OUTER_D
print(f"Diagram scale: {DS:.2f} px/mm")

# ── Diagram ruler helpers ─────────────────────────────────────────────────────
DC_GOLD  = (180, 120,   0)
DC_TEAL  = (  0, 120, 170)
DC_GREY  = ( 60,  60,  60)
DC_LGREY = (160, 160, 160)
DT = 9

def d_lbl(dd, xy, text, font, fg, bg=(250,250,250)):
    bb = dd.textbbox(xy, text, font=font, anchor="mm")
    p = 4
    dd.rectangle([bb[0]-p, bb[1]-p, bb[2]+p, bb[3]+p], fill=bg)
    dd.text(xy, text, fill=fg, font=font, anchor="mm")

def d_ruler_h(dd, x1, x2, y, label, color, above=True):
    dd.line([(x1,y),(x2,y)], fill=color, width=2)
    dd.line([(x1,y-DT),(x1,y+DT)], fill=color, width=2)
    dd.line([(x2,y-DT),(x2,y+DT)], fill=color, width=2)
    ly = y - DT - 15 if above else y + DT + 15
    d_lbl(dd, ((x1+x2)//2, ly), label, fnt_body, color)

def d_ruler_v(dd, x, y1, y2, label, color, right=True):
    dd.line([(x,y1),(x,y2)], fill=color, width=2)
    dd.line([(x-DT,y1),(x+DT,y1)], fill=color, width=2)
    dd.line([(x-DT,y2),(x+DT,y2)], fill=color, width=2)
    lx = x + DT + 60 if right else x - DT - 60
    d_lbl(dd, (lx, (y1+y2)//2), label, fnt_body, color)

# ── Rulers ────────────────────────────────────────────────────────────────────
ds_diag = int(DIAMOND_S_MM * DS)
dl_diag = int(DIAMOND_L_MM * DS)
bi_diag = int(WRIST_INNER_D * DS)

RULER_Y_BASE = TY + th_h + 22   # first ruler line below bracelet

# Bracelet outer diameter
d_ruler_h(dd, TX, TX+th_w, RULER_Y_BASE,
          f"Bracelet outer Ø  {BANGLE_OUTER_D:.1f} mm", DC_GREY, above=False)

# Wrist inner diameter (inset, 2nd line)
in_x1 = TX + (th_w - bi_diag)//2
in_x2 = in_x1 + bi_diag
d_ruler_h(dd, in_x1, in_x2, RULER_Y_BASE + 42,
          f"Wrist inner Ø  {WRIST_INNER_D:.1f} mm  (circ = {WRIST_CIRC_MM:.0f} mm)",
          DC_LGREY, above=False)

# Diamond positions on thumbnail: ~28% and ~72% x, ~78% y of thumb
ds_tx = TX + int(th_w * 0.28)
dl_tx = TX + int(th_w * 0.72)
gem_y_diag = TY + int(th_h * 0.78)

# Small diamond: H ruler below bracelet, V ruler on left side
H_GEM_Y = RULER_Y_BASE + 95
d_ruler_h(dd, ds_tx - ds_diag//2, ds_tx + ds_diag//2, H_GEM_Y,
          f"Ø {DIAMOND_S_MM} mm  (0.21 ct)", DC_GOLD, above=False)
d_ruler_v(dd, ds_tx - ds_diag//2 - 30,
          gem_y_diag - ds_diag//2, gem_y_diag + ds_diag//2,
          f"Ø {DIAMOND_S_MM} mm", DC_GOLD, right=False)

# Large diamond: H ruler below bracelet, V ruler on right side
d_ruler_h(dd, dl_tx - dl_diag//2, dl_tx + dl_diag//2, H_GEM_Y,
          f"Ø {DIAMOND_L_MM} mm  (0.42 ct)", DC_TEAL, above=False)
d_ruler_v(dd, dl_tx + dl_diag//2 + 30,
          gem_y_diag - dl_diag//2, gem_y_diag + dl_diag//2,
          f"Ø {DIAMOND_L_MM} mm", DC_TEAL, right=True)

# ── Scale bar ─────────────────────────────────────────────────────────────────
sb_px2 = int(10 * DS)
sbx2 = DIAG_W - sb_px2 - 30
sby2 = H_GEM_Y + 55
dd.line([(sbx2,sby2),(sbx2+sb_px2,sby2)], fill=(80,80,80), width=2)
dd.line([(sbx2,sby2-6),(sbx2,sby2+6)], fill=(80,80,80), width=2)
dd.line([(sbx2+sb_px2,sby2-6),(sbx2+sb_px2,sby2+6)], fill=(80,80,80), width=2)
dd.text((sbx2+sb_px2//2, sby2-11), "← 10 mm →", fill=(80,80,80), font=fnt_tiny, anchor="mb")

# ── Gem specification table ───────────────────────────────────────────────────
TABLE_Y = H_GEM_Y + 90
specs = [
    ("Stone",     "Shape",  "Size (mm)",                 "Weight",   "Colour"),
    ("Diamond 1", "Round",  f"Ø {DIAMOND_S_MM}×{DIAMOND_S_MM}", "0.21 ct", "D-F  VVS"),
    ("Diamond 2", "Round",  f"Ø {DIAMOND_L_MM}×{DIAMOND_L_MM}", "0.42 ct", "D-F  VVS"),
    ("Total",     "",       "",                           "0.63 ct",  ""),
]
COL_X = [40, 160, 310, 490, 620]
ROW_H = 28

# Table header background
dd.rectangle([30, TABLE_Y-4, DIAG_W-30, TABLE_Y+ROW_H-4],
             fill=(230,230,230))

for ri, row in enumerate(specs):
    for ci, cell in enumerate(row):
        if ci >= len(COL_X): break
        is_hdr = (ri == 0)
        col_map = {0: DC_GREY, 1: DC_GREY, 2: DC_GREY,
                   3: DC_GREY, 4: DC_GREY}
        # highlight stone colours
        if ri > 0 and ci == 0:
            fg = DC_GOLD if ri==1 else DC_TEAL
        elif is_hdr:
            fg = (80,80,80)
        else:
            fg = (40,40,40)
        font = fnt_bold if is_hdr else fnt_body
        dd.text((COL_X[ci], TABLE_Y + ri*ROW_H), cell, fill=fg, font=font)
    if ri == 0:
        # Underline header
        dd.line([(30, TABLE_Y+ROW_H-5),(DIAG_W-30, TABLE_Y+ROW_H-5)],
                fill=(180,180,180), width=1)

# Separator before Total row
if len(specs) > 1:
    sep_y = TABLE_Y + (len(specs)-1)*ROW_H - 5
    dd.line([(30, sep_y),(DIAG_W-30, sep_y)], fill=(180,180,180), width=1)

# Footer
dd.text((DIAG_W//2, DIAG_H - 20), "All measurements in millimetres  ·  Wrist size: 135 mm circumference",
        fill=(150,150,150), font=fnt_tiny, anchor="mb")

out2 = os.path.join(OUT_DIR, "DPBN0001_measurement_diagram.png")
diag.save(out2)
print(f"Saved diagram    : {out2}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
── Measurement summary ─────────────────────────────────────────
  Wrist circumference  : {WRIST_CIRC_MM} mm
  Wrist inner diameter : {WRIST_INNER_D:.2f} mm
  Bracelet outer diam  : {BANGLE_OUTER_D:.2f} mm
  Diamond small Ø      : {DIAMOND_S_MM} mm → {d_s_px} px in model image
  Diamond large Ø      : {DIAMOND_L_MM} mm → {d_l_px} px in model image
  Image scale          : {px_per_mm:.2f} px/mm
────────────────────────────────────────────────────────────────""")
