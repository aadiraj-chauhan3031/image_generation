#!/usr/bin/env python3
"""
DPBN0002 pipeline: generate 3 bracelet model images, measure pear & emerald
diamonds with Gemini Vision, produce a v2 comparison image for each.

Product: slim white-gold open-cuff bangle
  D1 — pear-cut diamond: 5.6 × 3.6mm (spec: measure longest dimension)
  D2 — emerald-cut diamond: 4.8 × 3.2mm (spec: measure longest dimension)

Usage:
  export GEMINI_API_KEY="your-key"
  python3 gen_compare_dpbn0002.py
"""

import base64, json, os, sys, time, math, io, re, urllib.request, urllib.error
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DIR  = "/Users/aadirajchauhan/Desktop/Practice coding/codeVector_work/image_generation"
OUT  = f"{DIR}/scale_meaurement/output/dpbn0002"
KEY  = os.environ.get("GEMINI_API_KEY")
API  = "https://generativelanguage.googleapis.com/v1beta"

SPEC_D1_MM = 5.6   # pear-cut, tip-to-tip length
SPEC_D2_MM = 4.8   # emerald-cut, longest length
CAD_GAP_MM = 17.9  # centre-to-centre gap measured from DPBN0002/FRONT.jpg

PROD_IMG = f"{DIR}/DPBN0002/TOP.jpg"
REF_IMGS = [
    f"{DIR}/scale_meaurement/model/close-up_shot.png",
]

GEN_PROMPT = (
    "Generate a photorealistic close-up jewellery editorial photograph of a slim white-gold "
    "open-cuff bangle bracelet worn on a woman's wrist. The bracelet has a prong-set "
    "PEAR-CUT diamond (teardrop shape, 5.6mm long) at one open end and a prong-set "
    "EMERALD-CUT diamond (rectangular step-cut, 4.8mm long) at the other open end. "
    "Use the composition and lighting of the reference photo: warm skin tone, intimate close-up, "
    "shallow depth of field, natural light. Both diamonds must be FULLY VISIBLE, sharply in "
    "focus, with the complete face of each stone clearly showing. Shot from a slightly elevated "
    "side angle so both the pear-cut and the emerald-cut stones are simultaneously seen face-on. "
    "The pear stone should clearly display its teardrop outline; the emerald stone its rectangular "
    "step-cut facets. Professional jewellery photography, white-gold metal."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_call(endpoint, body, timeout=300):
    req = urllib.request.Request(
        f"{API}/{endpoint}",
        headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
        data=json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")


def file_to_b64(path):
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode()


def pil_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


# ─── Step 1: Generate 3 images ────────────────────────────────────────────────

def generate_images(n=3):
    parts = []
    for ref in REF_IMGS:
        if os.path.isfile(ref):
            mime, data = file_to_b64(ref)
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
            print(f"  Using reference: {os.path.basename(ref)}")
    parts.append({"text": GEN_PROMPT})

    cfg = {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "4:3"}}
    paths = []

    for i in range(1, n + 1):
        print(f"\nGenerating {i}/{n} ...")
        try:
            data = api_call(
                "models/gemini-3-pro-image:generateContent",
                {"contents": [{"parts": parts}], "generationConfig": cfg},
            )
            for c in data.get("candidates", []):
                for part in c.get("content", {}).get("parts", []):
                    if part.get("thought"):
                        continue
                    blob = part.get("inlineData") or part.get("inline_data")
                    if blob:
                        ext   = (blob.get("mimeType") or "image/jpeg").split("/")[-1]
                        fname = f"dpbn0002_{time.strftime('%H%M%S')}_{i}.{ext}"
                        fpath = os.path.join(OUT, fname)
                        with open(fpath, "wb") as f:
                            f.write(base64.b64decode(blob["data"]))
                        print(f"  Saved: {fpath}")
                        paths.append(fpath)
        except Exception as e:
            print(f"  Generation {i} failed: {e}")
        time.sleep(1)

    return paths


# ─── Step 2: Estimate scale (px/mm) from bright cluster gap ──────────────────

def estimate_scale(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    arr  = np.array(img, dtype=np.uint16)
    brt  = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]

    # Exclude skin pixels (warm tones) — keep only bright non-skin sparkle areas
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    # Skin: R-B > 40 and R > G > B pattern
    is_skin = (r - b > 40) & (r > g) & (g > b * 0.9)

    for threshold in [680, 630, 580]:
        bright = (brt > threshold) & ~is_skin
        bys, bxs = np.where(bright)
        if len(bxs) >= 20:
            break

    if len(bxs) < 20:
        return None, None, None

    sx       = np.sort(bxs)
    split_x  = int(sx[np.argmax(np.diff(sx))])
    lm, rm   = bxs <= split_x, bxs > split_x

    if lm.sum() < 5 or rm.sum() < 5:
        return None, None, None

    cx1, cy1 = int(bxs[lm].mean()), int(bys[lm].mean())
    cx2, cy2 = int(bxs[rm].mean()), int(bys[rm].mean())
    gap_px   = math.hypot(cx2 - cx1, cy2 - cy1)

    if gap_px < 30:
        return None, None, None

    px_per_mm = gap_px / CAD_GAP_MM
    return px_per_mm, (cx1, cy1), (cx2, cy2)


# ─── Step 3: Measure stones with Gemini Vision ───────────────────────────────

def measure_with_gemini(img_path, px_per_mm, c1, c2):
    img  = Image.open(img_path).convert("RGB")
    w, h = img.size
    ann  = img.copy()
    draw = ImageDraw.Draw(ann)
    fnt  = load_font(18)

    # Draw 5mm scale bar
    sb_mm = 5
    sb_px = max(int(sb_mm * px_per_mm), 20)
    sbx, sby = 20, h - 60
    draw.rectangle([sbx, sby - 1, sbx + sb_px, sby + 9], fill=(255, 255, 255))
    draw.rectangle([sbx - 1, sby - 2, sbx + sb_px + 1, sby + 10], outline=(0, 0, 0), width=1)
    draw.text((sbx, sby + 13), f"◀ {sb_mm}mm ▶", fill=(255, 255, 255), font=fnt)

    # Number the diamonds with red dots
    for (cx, cy, label) in [(c1[0], c1[1], "1"), (c2[0], c2[1], "2")]:
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 50, 50))
        draw.text((cx + 10, cy - 12), label, fill=(255, 50, 50), font=fnt)

    # Resize for Vision API: cap at 1024px wide to avoid timeouts
    max_w = 1024
    if ann.width > max_w:
        ratio = max_w / ann.width
        ann_send = ann.resize((max_w, int(ann.height * ratio)), Image.LANCZOS)
    else:
        ann_send = ann
    b64 = pil_to_b64(ann_send)

    prompt = (
        f"This is a close-up photo of a white-gold open-cuff bracelet worn on a wrist. "
        f"Diamond '1' (red dot) is a PEAR-CUT stone (teardrop shape) at one end. "
        f"Diamond '2' (red dot) is an EMERALD-CUT stone (rectangular) at the other end. "
        f"There is a white scale bar at the bottom-left = exactly {sb_mm}mm. "
        f"Using ONLY that scale bar as your ruler: measure the LONGEST dimension of each stone "
        f"(tip-to-tip for the pear, long side for the emerald) — that is the longest edge-to-edge "
        f"dimension of the stone GIRDLE only, excluding all metal prong tips. "
        f"Measure independently for each stone. "
        f"Reply ONLY: D1=X.XX D2=X.XX  (mm, two decimal places, nothing else)"
    )

    body = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.1},
    }

    try:
        resp = api_call("models/gemini-3.8-flash:generateContent", body, timeout=120)
    except Exception as e:
        print(f"  Vision API error: {e}")
        return None, None

    text = "".join(
        part["text"]
        for c in resp.get("candidates", [])
        for part in c.get("content", {}).get("parts", [])
        if "text" in part
    ).strip()

    print(f"  Gemini says: {text}")

    m1 = re.search(r"D1\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    m2 = re.search(r"D2\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    return (float(m1.group(1)) if m1 else None,
            float(m2.group(1)) if m2 else None)


# ─── Step 4: Build comparison image in v2 layout ─────────────────────────────

def make_comparison(gen_path, d1_mm, d2_mm, px_per_mm, c1, c2):
    gen  = Image.open(gen_path).convert("RGB")
    gw, gh = gen.size
    fname  = os.path.basename(gen_path)

    diff1 = (d1_mm - SPEC_D1_MM) / SPEC_D1_MM * 100  # signed
    diff2 = (d2_mm - SPEC_D2_MM) / SPEC_D2_MM * 100

    fnt_hdr  = load_font(20)
    fnt_body = load_font(18)
    fnt_sm   = load_font(16)

    # ── Annotate model image ───────────────────────────────────────────────────
    ann  = gen.copy()
    drw  = ImageDraw.Draw(ann)
    BLUE = (60, 180, 255)

    # Use longest-dimension as the circle radius
    r1_px = max(int(d1_mm / 2 * px_per_mm), 8)
    r2_px = max(int(d2_mm / 2 * px_per_mm), 8)

    for (cx, cy), r, d_mm, diff in [
        (c1, r1_px, d1_mm, diff1),
        (c2, r2_px, d2_mm, diff2),
    ]:
        drw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BLUE, width=3)
        lx = cx + int(r * 1.3) + 8
        ly = cy - int(r * 0.9)
        drw.line([(cx + int(r * 0.7), cy - int(r * 0.7)), (lx, ly)], fill=BLUE, width=2)
        sign = "+" if diff >= 0 else ""
        label = f"gen ≈{d_mm:.2f}mm  ({sign}{diff:.0f}%)"
        bb  = drw.textbbox((0, 0), label, font=fnt_body)
        bw  = bb[2] - bb[0]; bh = bb[3] - bb[1]
        drw.rectangle([lx + 4, ly - bh // 2 - 5, lx + 4 + bw + 8, ly + bh // 2 + 5],
                      fill=(5, 20, 40))
        drw.text((lx + 8, ly), label, fill=(255, 255, 255), font=fnt_body, anchor="lm")

    # 5mm scale bar
    sb_mm = 5; sb_px = max(int(sb_mm * px_per_mm), 20)
    sbx, sby = 20, gh - 50
    drw.rectangle([sbx, sby, sbx + sb_px, sby + 6], fill=(255, 255, 255))
    drw.text((sbx, sby + 10), f"{sb_mm}mm", fill=(255, 255, 255), font=fnt_sm)

    # Header
    drw.rectangle([(0, 0), (gw, 42)], fill=(10, 10, 40))
    drw.text((14, 11), f"AI GENERATED  —  {fname}", fill=(100, 180, 255), font=fnt_hdr)

    # ── Canvas layout ──────────────────────────────────────────────────────────
    LEFT_W  = 620
    PANEL_H = 760
    BAR_H   = 110

    right_scale = PANEL_H / gh
    RIGHT_W     = int(gw * right_scale)
    ann_fitted  = ann.resize((RIGHT_W, PANEL_H), Image.LANCZOS)

    TOTAL_W = LEFT_W + RIGHT_W
    TOTAL_H = PANEL_H + BAR_H

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), (12, 12, 12))
    cd     = ImageDraw.Draw(canvas)

    # Left header
    cd.rectangle([0, 0, LEFT_W, 42], fill=(28, 28, 28))
    cd.text((12, 11), "SPEC  —  DPBN0002.3dm  (CAD source)", fill=(220, 180, 60), font=fnt_hdr)

    # Product image (TOP.jpg) in left panel
    prod     = Image.open(PROD_IMG).convert("RGB")
    pw, ph   = prod.size
    avail_w  = LEFT_W - 20
    avail_h  = PANEL_H - 42 - 20
    sc_p     = min(avail_w / pw, avail_h / ph)
    pnw, pnh = int(pw * sc_p), int(ph * sc_p)
    prod_th  = prod.resize((pnw, pnh), Image.LANCZOS)
    px0      = (LEFT_W - pnw) // 2
    py0      = 42 + (avail_h - pnh) // 2
    canvas.paste(prod_th, (px0, py0))

    # Spec circles on product
    # px_per_mm in the scaled product thumbnail
    prod_pxmm = pnw / 60.0

    # D1 pear: draw ellipse (5.6×3.6mm); D2 emerald: draw ellipse (4.8×3.2mm)
    # Relative positions from visual inspection of TOP.jpg (1200×1200)
    # Pear centre: ~26.7% x, ~50% y; Emerald centre: ~62% x, ~50% y
    pd1x = px0 + int(pnw * 0.267); pd1y = py0 + int(pnh * 0.500)
    pd2x = px0 + int(pnw * 0.620); pd2y = py0 + int(pnh * 0.500)

    RED = (220, 50, 50)

    # Pear-cut ellipse (5.6mm long × 3.6mm wide)
    r1lx = max(int(SPEC_D1_MM / 2 * prod_pxmm), 5)   # half-length
    r1wy = max(int(3.6 / 2 * prod_pxmm), 4)            # half-width
    cd.ellipse([pd1x - r1lx, pd1y - r1wy, pd1x + r1lx, pd1y + r1wy], outline=RED, width=3)
    cd.text((pd1x - r1lx, pd1y + r1wy + 4), f"D1={SPEC_D1_MM}mm (pear)", fill=RED, font=fnt_sm)

    # Emerald-cut rectangle (4.8mm long × 3.2mm wide)
    r2lx = max(int(SPEC_D2_MM / 2 * prod_pxmm), 5)   # half-length
    r2wy = max(int(3.2 / 2 * prod_pxmm), 4)            # half-width
    cd.rectangle([pd2x - r2lx, pd2y - r2wy, pd2x + r2lx, pd2y + r2wy], outline=RED, width=3)
    cd.text((pd2x - r2lx, pd2y + r2wy + 4), f"D2={SPEC_D2_MM}mm (emerald)", fill=RED, font=fnt_sm)

    # 10mm scale bar on left panel
    sb10 = max(int(10 * prod_pxmm), 1)
    cd.rectangle([20, PANEL_H - 32, 20 + sb10, PANEL_H - 26], fill=(40, 40, 40))
    cd.text((20, PANEL_H - 22), "10mm", fill=(80, 80, 80), font=fnt_sm)

    # Divider
    cd.line([(LEFT_W, 0), (LEFT_W, PANEL_H)], fill=(55, 55, 55), width=2)

    # Paste model panel
    canvas.paste(ann_fitted, (LEFT_W, 0))

    # ── Bottom comparison bar ──────────────────────────────────────────────────
    ty = PANEL_H + 10
    cd.text((12, ty), "DIAMOND SIZE COMPARISON — DPBN0002", fill=(140, 140, 140), font=fnt_sm)

    COL_L  = [12,   220,  LEFT_W + 30, LEFT_W + 230, LEFT_W + 390]
    for cx_, h in zip(COL_L, ["", "SPEC (CAD)", "", "AI GENERATED", "Δ DIFF"]):
        cd.text((cx_, ty), h, fill=(110, 110, 110), font=fnt_sm)

    for ri, (name, spec_v, gen_v, diff) in enumerate([
        ("D1 — pear-cut (5.6×3.6mm)",    f"{SPEC_D1_MM}mm", f"{d1_mm:.2f}mm", diff1),
        ("D2 — emerald-cut (4.8×3.2mm)", f"{SPEC_D2_MM}mm", f"{d2_mm:.2f}mm", diff2),
    ]):
        ry   = ty + 30 + ri * 30
        sign = "+" if diff >= 0 else ""
        col  = (80, 210, 80) if abs(diff) <= 10 else (220, 100, 60)
        cd.text((COL_L[0], ry), name,                     fill=(180, 180, 180), font=fnt_body)
        cd.text((COL_L[1], ry), spec_v,                   fill=(220, 180, 60),  font=fnt_body)
        cd.text((COL_L[3], ry), gen_v,                    fill=(100, 180, 255), font=fnt_body)
        cd.text((COL_L[4], ry), f"□ {sign}{diff:.0f}%",  fill=col,             font=fnt_body)

    out_name = f"comparison_dpbn0002_{os.path.splitext(fname)[0]}.jpg"
    out_path = os.path.join(OUT, out_name)
    canvas.save(out_path, quality=96)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not KEY:
        sys.exit('Set GEMINI_API_KEY first:  export GEMINI_API_KEY="..."')

    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("DPBN0002 PIPELINE")
    print(f"  D1 (pear):    {SPEC_D1_MM}mm length")
    print(f"  D2 (emerald): {SPEC_D2_MM}mm length")
    print(f"  CAD gap:      {CAD_GAP_MM}mm")
    print("=" * 60)

    print("\nSTEP 1 — Generating 3 model images")
    print("-" * 40)
    gen_paths = generate_images(3)

    if not gen_paths:
        sys.exit("No images generated.")

    print("\nSTEP 2 — Measuring & comparing each image")
    print("-" * 40)

    results = []
    for path in gen_paths:
        print(f"\n--- {os.path.basename(path)} ---")

        px_per_mm, c1, c2 = estimate_scale(path)
        if px_per_mm is None:
            print("  Could not estimate scale — skipping")
            continue
        print(f"  Scale: {px_per_mm:.2f} px/mm  |  "
              f"D1 centre {c1}  D2 centre {c2}")

        d1_mm, d2_mm = measure_with_gemini(path, px_per_mm, c1, c2)
        if d1_mm is None or d2_mm is None:
            print("  Measurement failed — skipping")
            continue

        diff1 = abs(d1_mm - SPEC_D1_MM) / SPEC_D1_MM * 100
        diff2 = abs(d2_mm - SPEC_D2_MM) / SPEC_D2_MM * 100
        ok    = "✓" if diff1 <= 10 and diff2 <= 10 else "✗"
        print(f"  {ok} D1={d1_mm:.2f}mm (Δ{diff1:.0f}%)  "
              f"D2={d2_mm:.2f}mm (Δ{diff2:.0f}%)")

        comp_path = make_comparison(path, d1_mm, d2_mm, px_per_mm, c1, c2)
        print(f"  Comparison saved: {comp_path}")

        results.append({
            "path": path, "comp": comp_path,
            "d1": d1_mm, "d2": d2_mm,
            "err1": diff1, "err2": diff2,
        })

    if not results:
        print("\nNo results produced.")
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        ok = "✓" if r["err1"] <= 10 and r["err2"] <= 10 else "✗"
        print(f"{ok} {os.path.basename(r['path'])}: "
              f"D1={r['d1']:.2f}mm (Δ{r['err1']:.0f}%)  "
              f"D2={r['d2']:.2f}mm (Δ{r['err2']:.0f}%)")

    best = min(results, key=lambda r: r["err1"] + r["err2"])
    print(f"\nBest: {os.path.basename(best['path'])}")
    print(f"      D1={best['d1']:.2f}mm (Δ{best['err1']:.0f}%)  "
          f"D2={best['d2']:.2f}mm (Δ{best['err2']:.0f}%)")
    print(f"      Comparison: {best['comp']}")


if __name__ == "__main__":
    main()
