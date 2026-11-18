#!/usr/bin/env python3
"""
Lifestyle image pipeline for diamond bangle products.

Usage:
  python3 pipeline.py <SKU> \\
    --refs /path/FRONT.jpg /path/TOP.jpg /path/3D.jpg \\
    --hero /path/TOP.jpg \\
    --diamonds "pear:4.0x5.9x2.6" "emerald:3.2x4.4x1.9"

  --diamonds format:  cut:WIDTHxHEIGHTxDEPTH  (all in mm)
                      e.g.  "baguette:3.0x6.0x1.9"
  --hero              defaults to first --refs image
  --batch             number of lifestyle images to generate (default 3)
  --bangle-mm         bangle outer diameter for ruler scale (default 61)

Outputs:  output/<SKU>/
            prompt.txt
            lifestyle_1.jpeg ... lifestyle_N.jpeg
            split.jpg          (hero | lifestyle_1)
            annotated.jpg      (split + mm rulers)

Requires: GEMINI_API_KEY env var set.
"""

import argparse, base64, json, mimetypes, os, sys, time
import urllib.request, urllib.error
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_REF  = os.path.join(SCRIPT_DIR, "..", "scale_meaurement", "model", "close-up_shot.png")
API        = "https://generativelanguage.googleapis.com/v1beta"
GEM_MODEL  = "nano-banana-pro-preview"
FONT_PATH  = "/System/Library/Fonts/Helvetica.ttc"
GAP_PX     = 20   # gap between panels in split image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sku",             help="Product SKU, e.g. DPBN0004")
    p.add_argument("--refs",          nargs="+", required=True,
                   help="Product reference images (fed to AI)")
    p.add_argument("--hero",          default=None,
                   help="Product image used in the split (defaults to first --ref)")
    p.add_argument("--diamonds",      nargs="+", required=True,
                   help="Diamond specs: cut:WxHxD  e.g. 'pear:4.0x5.9x2.6'")
    p.add_argument("--batch",         type=int, default=3,
                   help="Number of lifestyle images to generate")
    p.add_argument("--bangle-mm",     type=float, default=61.0,
                   help="Bangle outer diameter in mm (for ruler scale, default 61)")
    p.add_argument("--model",         default=GEM_MODEL,
                   help=f"Gemini model (default: {GEM_MODEL})")
    p.add_argument("--product-desc",  default=None,
                   help="Full product description (overrides auto-built open-cuff template)")
    return p.parse_args()


def parse_diamonds(specs):
    """Parse ['pear:4.0x5.9x2.6', 'emerald:3.2x4.4x1.9'] → list of dicts."""
    result = []
    for s in specs:
        cut, dims = s.split(":")
        parts = dims.split("x")
        result.append({
            "cut":   cut.strip().lower(),
            "width": float(parts[0]),
            "height": float(parts[1]),
            "depth":  float(parts[2]) if len(parts) > 2 else 0.0,
        })
    return result


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(diamonds, product_desc=None):
    if product_desc:
        product_block = product_desc.strip()
    else:
        lines = []
        for d in diamonds:
            lines.append(
                f"  - {d['cut'].title()}-cut diamond: {d['width']} mm wide x {d['height']} mm tall"
                + (f" x {d['depth']} mm deep" if d['depth'] else "")
                + " -- keep it SMALL and delicate"
            )
        product_block = (
            "Open-ended bangle in polished white gold / platinum:\n"
            "- Slim, perfectly round wire cuff with a gap at the top\n"
            "- At each open end sits one diamond in a minimal 4-prong setting:\n"
            + "\n".join(lines)
            + "\n- The diamonds are small accent stones -- do NOT exaggerate their size"
        )

    return f"""You are given two sets of images:
1. FIRST IMAGE: A lifestyle photograph of a model's wrist wearing a gold bracelet with coloured gemstones.
2. REMAINING IMAGES: Multiple perspectives of a different bracelet -- our product.

TASK: Recreate the first image exactly, but replace the bracelet on the model's wrist with our product bracelet shown in the remaining reference images.

OUR PRODUCT:
{product_block}

PRESERVE from the original model image:
- Wrist pose, angle, skin tone, and overall composition
- Warm editorial lighting, shadows, and bokeh background

REPLACE only the existing bracelet with our product. It should sit naturally on the wrist in the same position.

CRITICAL — stone sizing: render all diamonds true to their stated mm dimensions. Do NOT enlarge or exaggerate any stone. The stones should look delicate and proportionate on the wrist, not oversized.

Style: photorealistic high-end luxury jewellery editorial, 4K, sharp focus on diamonds and metal."""


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
def api_call(url, body, key):
    req = urllib.request.Request(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        data=json.dumps(body).encode()
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  API error {e.code}: {e.read().decode()[:300]}")
        return None


def generate_one(prompt, ref_paths, run_num, out_dir, model, key, run_ts=None):
    parts = []
    for path in ref_paths:
        if not os.path.isfile(path):
            print(f"  Warning: image not found: {path}")
            continue
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            parts.append({"inlineData": {"mimeType": mime,
                                         "data": base64.b64encode(f.read()).decode()}})
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "4:5"}
        }
    }
    data = api_call(f"{API}/models/{model}:generateContent", body, key)
    if not data:
        return None

    for c in data.get("candidates", []):
        for part in c.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            if "text" in part:
                print(f"  Model: {part['text'].strip()[:100]}")
            blob = part.get("inlineData") or part.get("inline_data")
            if blob:
                ext  = (blob.get("mimeType") or blob.get("mime_type") or
                        "image/jpeg").split("/")[-1]
                ts   = run_ts or time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir,
                                    f"lifestyle_{ts}_{run_num}.{ext}")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                print(f"  Saved: {path}")
                return path
    print("  No image returned.")
    return None


# ---------------------------------------------------------------------------
# Split image
# ---------------------------------------------------------------------------
def make_split(hero_path, lifestyle_path, out_path):
    left  = Image.open(hero_path).convert("RGB")
    right = Image.open(lifestyle_path).convert("RGB")
    target_h = max(left.height, right.height)

    def to_h(img, h):
        r = h / img.height
        return img.resize((int(img.width * r), h), Image.LANCZOS)

    left  = to_h(left,  target_h)
    right = to_h(right, target_h)

    canvas = Image.new("RGB", (left.width + GAP_PX + right.width, target_h),
                       (255, 255, 255))
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width + GAP_PX, 0))
    canvas.save(out_path, quality=95)
    return left.width, right.width   # panel widths (used by annotator)


# ---------------------------------------------------------------------------
# Diamond detection helpers
# ---------------------------------------------------------------------------
def detect_diamonds_product(img_array, n=2):
    """Find n diamond clusters in a white-background product image."""
    gray = np.mean(img_array, axis=2)
    mask = gray < 220
    h, w = mask.shape

    # Focus on lower 55 % where gap/diamonds are in TOP view
    y_lo = int(h * 0.45)
    sub  = mask[y_lo:, :]
    col_counts = sub.sum(axis=0)
    hot  = np.where(col_counts > 3)[0]
    if not len(hot):
        return []

    gaps = np.where(np.diff(hot) > 20)[0]
    clusters, prev = [], 0
    for g in gaps:
        clusters.append(hot[prev:g+1]); prev = g+1
    clusters.append(hot[prev:])
    clusters = sorted([c for c in clusters if len(c) > 5],
                      key=lambda c: col_counts[c].sum(), reverse=True)[:n]
    clusters.sort(key=lambda c: c[0])   # left → right order

    result = []
    for cl in clusters:
        x0, x1 = int(cl[0]), int(cl[-1])
        sub2 = sub[:, x0:x1+1]
        ry = np.where(sub2.any(axis=1))[0]
        y0, y1 = y_lo + int(ry[0]), y_lo + int(ry[-1])
        result.append({"cx": (x0+x1)//2, "cy": (y0+y1)//2, "x1": x1, "y1": y1})
    return result


def detect_diamonds_lifestyle(img_array, n=2):
    """Find n diamond clusters in a lifestyle wrist image."""
    arr = img_array.astype(float)
    brightness  = arr.mean(axis=2)
    saturation  = arr.max(axis=2) - arr.min(axis=2)
    h, _        = brightness.shape
    y_lo, y_hi  = int(h * 0.35), int(h * 0.80)
    zone        = (brightness > 155) & (saturation < 45)
    sub         = zone[y_lo:y_hi, :]
    col_counts  = sub.sum(axis=0)
    hot         = np.where(col_counts > 5)[0]
    if not len(hot):
        return []

    gaps = np.where(np.diff(hot) > 25)[0]
    clusters, prev = [], 0
    for g in gaps:
        clusters.append(hot[prev:g+1]); prev = g+1
    clusters.append(hot[prev:])
    clusters = sorted([c for c in clusters if len(c) > 5],
                      key=lambda c: col_counts[c].sum(), reverse=True)[:n]
    clusters.sort(key=lambda c: c[0])

    result = []
    for cl in clusters:
        x0, x1 = int(cl[0]), int(cl[-1])
        sub2 = sub[:, x0:x1+1]
        ry = np.where(sub2.any(axis=1))[0]
        y0, y1 = y_lo + int(ry[0]), y_lo + int(ry[-1])
        result.append({"cx": (x0+x1)//2, "cy": (y0+y1)//2, "x1": x1, "y1": y1})
    return result


# ---------------------------------------------------------------------------
# Ruler drawing
# ---------------------------------------------------------------------------
def outline_text(draw, xy, text, font, fill, outline=(0,0,0), stroke=2):
    x, y = xy
    for dx in range(-stroke, stroke+1):
        for dy in range(-stroke, stroke+1):
            if dx or dy:
                draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)


def ruler_v(draw, cx, cy, h_px, label, color, font, offset_x=18):
    tick = 10
    x  = cx + offset_x
    y0 = int(cy - h_px / 2)
    y1 = int(cy + h_px / 2)
    draw.line([(x, y0), (x, y1)], fill=color, width=2)
    draw.line([(x-tick, y0), (x+tick, y0)], fill=color, width=2)
    draw.line([(x-tick, y1), (x+tick, y1)], fill=color, width=2)
    bb = font.getbbox(label)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    outline_text(draw, (x + tick + 4, int(cy - th/2)), label, font, color)


def ruler_h(draw, cx, cy, w_px, label, color, font, offset_y=18):
    tick = 10
    y  = cy + offset_y
    x0 = int(cx - w_px / 2)
    x1 = int(cx + w_px / 2)
    draw.line([(x0, y), (x1, y)], fill=color, width=2)
    draw.line([(x0, y-tick), (x0, y+tick)], fill=color, width=2)
    draw.line([(x1, y-tick), (x1, y+tick)], fill=color, width=2)
    bb = font.getbbox(label)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    outline_text(draw, (int(cx - tw/2), y + tick + 2), label, font, color)


def annotate(split_path, out_path, diamonds, bangle_mm,
             hero_w, life_w, hero_orig_w, life_orig_w, life_orig_h):
    """Draw mm rulers on both halves of the split image."""
    img  = Image.open(split_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    split_h = img.height

    try:
        font_lg = ImageFont.truetype(FONT_PATH, 26)
        font_sm = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        font_lg = font_sm = ImageFont.load_default()

    # ---- Scales ----
    prod_pxmm = hero_w / bangle_mm   # product panel px/mm

    # Lifestyle scale via diamond-to-diamond gap reference
    life_arr = np.array(Image.open(split_path).convert("RGB"))[
        :, hero_w + GAP_PX:, :]
    life_diamonds = detect_diamonds_lifestyle(life_arr, n=len(diamonds))

    prod_arr = np.array(Image.open(split_path).convert("RGB"))[:, :hero_w, :]
    prod_diamonds = detect_diamonds_product(prod_arr, n=len(diamonds))

    # Estimate lifestyle px/mm from gap if 2+ diamonds detected on each side
    life_pxmm = prod_pxmm * 0.35   # fallback: assume typical wrist-shot zoom
    if len(prod_diamonds) >= 2 and len(life_diamonds) >= 2:
        prod_gap = abs(prod_diamonds[1]["cx"] - prod_diamonds[0]["cx"])
        life_gap = abs(life_diamonds[1]["cx"] - life_diamonds[0]["cx"])
        if prod_gap > 0:
            life_pxmm = life_gap / (prod_gap / prod_pxmm)

    # ---- Annotate product side (LEFT) ----
    PROD_COLOR = (220, 20, 60)   # crimson
    for i, gem in enumerate(diamonds):
        if i >= len(prod_diamonds):
            break
        d   = prod_diamonds[i]
        cx, cy, x1, y1 = d["cx"], d["cy"], d["x1"], d["y1"]
        w_px = round(gem["width"]  * prod_pxmm)
        h_px = round(gem["height"] * prod_pxmm)
        ruler_v(draw, x1, cy, h_px, f"{gem['height']} mm", PROD_COLOR, font_lg, offset_x=20)
        ruler_h(draw, cx, y1, w_px, f"{gem['width']} mm",  PROD_COLOR, font_lg, offset_y=16)

    # ---- Annotate lifestyle side (RIGHT) ----
    OX = hero_w + GAP_PX
    LIFE_COLOR = (255, 220, 0)   # golden yellow
    for i, gem in enumerate(diamonds):
        if i >= len(life_diamonds):
            break
        d   = life_diamonds[i]
        cx  = d["cx"] + OX
        cy  = d["cy"]
        x1  = d["x1"] + OX
        y1  = d["y1"]
        w_px = round(gem["width"]  * life_pxmm)
        h_px = round(gem["height"] * life_pxmm)
        ruler_v(draw, x1, cy, h_px, f"{gem['height']} mm", LIFE_COLOR, font_sm, offset_x=16)
        ruler_h(draw, cx, y1, w_px, f"{gem['width']} mm",  LIFE_COLOR, font_sm, offset_y=12)

    img.save(out_path, quality=95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args    = parse_args()
    key     = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first.")

    diamonds = parse_diamonds(args.diamonds)
    refs     = [os.path.abspath(r) for r in args.refs]
    if args.hero:
        hero = os.path.abspath(args.hero)
    else:
        # prefer TOP.jpg among refs, fall back to first ref
        top_match = next((r for r in refs if os.path.basename(r).upper() == "TOP.JPG"), None)
        hero = top_match or refs[0]
    model_ref = os.path.abspath(MODEL_REF)

    out_dir = os.path.join(SCRIPT_DIR, "output", args.sku)
    os.makedirs(out_dir, exist_ok=True)

    run_ts = time.strftime("%Y%m%d_%H%M%S")   # unique stamp per run

    # ---- Build & save prompt ------------------------------------------------
    prompt = build_prompt(diamonds, product_desc=args.product_desc)
    with open(os.path.join(out_dir, f"prompt_{run_ts}.txt"), "w") as f:
        f.write(prompt)
    print(f"\nSKU    : {args.sku}")
    print(f"Run    : {run_ts}")
    print(f"Model  : {args.model}")
    print(f"Batch  : {args.batch}")
    print(f"Output : {out_dir}\n")

    # ---- Generate lifestyle images ------------------------------------------
    all_refs   = [model_ref] + refs
    first_life = None
    for i in range(1, args.batch + 1):
        print(f"[{i}/{args.batch}] Generating lifestyle image...")
        path = generate_one(prompt, all_refs, i, out_dir, args.model, key,
                            run_ts=run_ts)
        if path and first_life is None:
            first_life = path
        if i < args.batch:
            time.sleep(2)

    if not first_life:
        sys.exit("No images were generated. Check API key and model access.")

    # ---- Split image --------------------------------------------------------
    split_path = os.path.join(out_dir, f"split_{run_ts}.jpg")
    print(f"\nCreating split image...")
    hero_img  = Image.open(hero)
    life_img  = Image.open(first_life)
    target_h  = max(hero_img.height, life_img.height)
    hero_w_resized = int(hero_img.width * target_h / hero_img.height)
    life_w_resized = int(life_img.width * target_h / life_img.height)
    make_split(hero, first_life, split_path)
    print(f"Saved : {split_path}")

    # ---- Annotated image ----------------------------------------------------
    ann_path = os.path.join(out_dir, f"annotated_{run_ts}.jpg")
    print(f"Adding rulers...")
    annotate(
        split_path, ann_path, diamonds, args.bangle_mm,
        hero_w=hero_w_resized,
        life_w=life_w_resized,
        hero_orig_w=hero_img.width,
        life_orig_w=life_img.width,
        life_orig_h=life_img.height,
    )
    print(f"Saved : {ann_path}")
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
