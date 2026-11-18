#!/usr/bin/env python3
"""
Bulk generate bracelet-swap images for DPBN0003 using nano-banana-pro-preview.
Swaps the bracelet in the model reference photo with DPBN0003 product.
Outputs go to ./output/
Run: python3 generate_dpbn0003.py
"""
import base64, json, mimetypes, os, sys, time, urllib.request, urllib.error

API   = "https://generativelanguage.googleapis.com/v1beta"
KEY   = os.environ.get("GEMINI_API_KEY")
MODEL = "nano-banana-pro-preview"
BATCH = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_ROOT  = os.path.dirname(SCRIPT_DIR)
BANGLE_ROOT = os.path.join(os.path.dirname(IMG_ROOT), "BANGLE")

MODEL_REF = os.path.join(IMG_ROOT, "scale_meaurement", "model", "close-up_shot.png")

PRODUCT_REFS = [
    os.path.join(BANGLE_ROOT, "DPBN0003", "FRONT.jpg"),
    os.path.join(BANGLE_ROOT, "DPBN0003", "TOP.jpg"),
    os.path.join(BANGLE_ROOT, "DPBN0003", "3D.jpg"),
    os.path.join(BANGLE_ROOT, "DPBN0003", "3 COLOR.jpg"),
]

PROMPT = """You are given two sets of images:
1. FIRST IMAGE: A lifestyle photograph of a model's wrist wearing a gold bracelet with green and blue gemstones.
2. REMAINING IMAGES: Multiple perspectives (front, top, 3D render, color render) of a different bracelet — our product DPBN0003.

TASK: Recreate the first image exactly, but replace the bracelet on the model's wrist with our DPBN0003 product bracelet shown in the remaining reference images.

OUR PRODUCT details (DPBN0003):
- Open-ended bangle: slim, perfectly round wire cuff in polished white gold / platinum with a gap at the top
- BOTH open ends have MATCHING straight baguette-cut diamonds in 4-prong settings
- Each baguette diamond: 3.0 mm wide × 6.0 mm long × 1.9 mm deep — slim rectangular stones, portrait orientation (taller than wide)
- The two diamonds are identical — same cut, same size, one on each end
- The settings are minimal, clean, and symmetrical
- Keep the diamonds their true small size — do NOT enlarge them

PRESERVE from the original model image:
- The model's wrist pose, angle, and position exactly as-is
- Skin tone, lighting direction, shadows, and warm bokeh background
- The overall editorial composition and framing

REPLACE only:
- The existing gold bracelet → swap with our white gold open bangle (DPBN0003)
- The bangle sits naturally on the wrist in the same position the original bracelet occupies
- Both matching baguette-diamond ends should be visible at the top gap

Result: a photorealistic high-end jewellery editorial image, 4K quality, sharp focus on the baguette diamonds and bangle wire, warm natural lighting preserved from the reference scene."""


def call(url, body):
    req = urllib.request.Request(
        url,
        headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
        data=json.dumps(body).encode()
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  API error {e.code}: {err[:300]}")
        return None


def generate(run_num):
    parts = []
    for path in [MODEL_REF] + PRODUCT_REFS:
        if not os.path.isfile(path):
            print(f"  Warning: ref not found: {path}")
            continue
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(f.read()).decode()}})

    parts.append({"text": PROMPT})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "4:5"}
        }
    }

    data = call(f"{API}/models/{MODEL}:generateContent", body)
    if not data:
        return False

    saved = False
    for c in data.get("candidates", []):
        for part in c.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            if "text" in part:
                print(f"  Model: {part['text'].strip()[:120]}")
            blob = part.get("inlineData") or part.get("inline_data")
            if blob:
                ext = (blob.get("mimeType") or blob.get("mime_type") or "image/jpeg").split("/")[-1]
                fname = os.path.join(OUT_DIR, f"dpbn0003_swap_{time.strftime('%H%M%S')}_{run_num}.{ext}")
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                print(f"  Saved: {fname}")
                saved = True

    if not saved:
        print("  No image returned.")
        if data and "promptFeedback" in data:
            print(" ", json.dumps(data["promptFeedback"], indent=2)[:300])
    return saved


if __name__ == "__main__":
    if not KEY:
        sys.exit("Set your key first:  export GEMINI_API_KEY='your-key'")

    print(f"Product: DPBN0003 — Twin Baguette Bangle")
    print(f"Model  : {MODEL}")
    print(f"Batch  : {BATCH} images")
    print(f"Output : {OUT_DIR}\n")

    for i in range(1, BATCH + 1):
        print(f"[{i}/{BATCH}] Generating...")
        generate(i)
        if i < BATCH:
            time.sleep(2)

    print("\nDone.")
