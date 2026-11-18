#!/usr/bin/env python3
"""
Bulk generate bracelet-on-wrist images using nano-banana-pro-preview.
Outputs go to ./output/
Run: python3 generate.py
"""
import base64, json, mimetypes, os, sys, time, urllib.request, urllib.error

API   = "https://generativelanguage.googleapis.com/v1beta"
KEY   = os.environ.get("GEMINI_API_KEY")
MODEL = "nano-banana-pro-preview"
BATCH = 3   # images per run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

ROOT = os.path.dirname(SCRIPT_DIR)

# Model reference image (scene to preserve)
MODEL_REF = os.path.join(ROOT, "scale_meaurement", "model", "close-up_shot.png")

# Product reference images (teach the model what to place)
PRODUCT_REFS = [
    os.path.join(ROOT, "DPBN0002", "FRONT.jpg"),
    os.path.join(ROOT, "DPBN0002", "TOP.jpg"),
    os.path.join(ROOT, "DPBN0002", "3D.jpg"),
    os.path.join(ROOT, "DPBN0002", "3 COLOR.jpg"),
]

PROMPT = """You are given two sets of images:
1. FIRST IMAGE: A lifestyle photograph of a model's wrist wearing a gold bracelet with green and blue gemstones.
2. REMAINING IMAGES: Multiple perspectives (front, top, 3D render, color render) of a different bracelet — our product.

TASK: Recreate the first image exactly, but replace the bracelet on the model's wrist with our product bracelet shown in the remaining reference images.

OUR PRODUCT details:
- Open-ended bangle: slim, round white gold / platinum wire cuff with a gap at the top
- Left open end: small pear-cut diamond (teardrop, 4-prong set), size 4.0 × 5.9 mm — keep it SMALL
- Right open end: small emerald-cut diamond (rectangular step-cut, 4-prong set), size 3.2 × 4.4 mm — keep it SMALL
- The diamonds are delicate accent stones, NOT large centrepieces

PRESERVE from the original model image:
- The model's wrist pose, angle, and position exactly
- Skin tone, lighting, shadows, and background atmosphere
- The warm, soft editorial lighting and bokeh
- Overall composition and framing

REPLACE only:
- The existing bracelet → swap with our white gold open bangle
- The bangle should sit naturally on the wrist in the same position the original bracelet occupies
- Both diamond ends should be visible at the top gap of the bangle

Result: a photorealistic high-end jewellery editorial image, 4K quality, sharp focus on the diamonds and bangle wire."""


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
    # Model reference image first
    for path in [MODEL_REF] + PRODUCT_REFS:
        if not os.path.isfile(path):
            print(f"  Warning: ref image not found: {path}")
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
                fname = os.path.join(OUT_DIR, f"swap_{time.strftime('%H%M%S')}_{run_num}.{ext}")
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
                print(f"  Saved: {fname}")
                saved = True

    if not saved:
        print("  No image returned for this call.")
        if "promptFeedback" in data:
            print(" ", json.dumps(data["promptFeedback"], indent=2)[:300])
    return saved


if __name__ == "__main__":
    if not KEY:
        sys.exit("Set your key first:  export GEMINI_API_KEY='your-key'")

    print(f"Model : {MODEL}")
    print(f"Batch : {BATCH} images")
    print(f"Output: {OUT_DIR}\n")

    for i in range(1, BATCH + 1):
        print(f"[{i}/{BATCH}] Generating...")
        generate(i)
        if i < BATCH:
            time.sleep(2)

    print("\nDone.")
