#!/usr/bin/env python3
"""
Generate images with Gemini from a prompt + optional reference photos.

Usage:
  python3 gen.py "your prompt" [photo1.jpg photo2.png ...]

Options (set before the command, optional):
  GEMINI_MODEL=gemini-2.5-flash-image   which image model to use
  ASPECT=4:5                            aspect ratio (1:1, 3:4, 4:5, 9:16, 16:9 ...)
  COUNT=3                               how many images to make

List the image models your key can use:
  python3 gen.py --models
"""
import base64, json, mimetypes, os, sys, time, urllib.request, urllib.error

API = "https://generativelanguage.googleapis.com/v1beta"
KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image")
ASPECT = os.environ.get("ASPECT")
COUNT = int(os.environ.get("COUNT", "1"))


def call(url, body=None):
    req = urllib.request.Request(url, headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"API error {e.code}:\n{e.read().decode()}")


def list_models():
    data = call(f"{API}/models?pageSize=1000")
    names = [m["name"].split("/")[-1] for m in data.get("models", [])]
    img = [n for n in names if "image" in n or "banana" in n]
    print("Image models available to your key:" if img else "No image models found. All models:")
    for n in img or names:
        print("  ", n)


def generate(prompt, photos, run):
    parts = []
    for p in photos:
        mime = mimetypes.guess_type(p)[0] or "image/jpeg"
        with open(p, "rb") as f:
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(f.read()).decode()}})
    parts.append({"text": prompt})
    cfg = {"responseModalities": ["TEXT", "IMAGE"]}
    if ASPECT:
        cfg["imageConfig"] = {"aspectRatio": ASPECT}

    data = call(f"{API}/models/{MODEL}:generateContent",
                {"contents": [{"parts": parts}], "generationConfig": cfg})

    saved = None
    for c in data.get("candidates", []):
        for part in c.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            if "text" in part:
                print("Model says:", part["text"].strip())
            blob = part.get("inlineData") or part.get("inline_data")
            if blob:
                ext = (blob.get("mimeType") or blob.get("mime_type") or "image/png").split("/")[-1]
                saved = f"output_{time.strftime('%H%M%S')}_{run}.{ext}"
                with open(saved, "wb") as f:
                    f.write(base64.b64decode(blob["data"]))
    if saved:
        print("Saved:", saved)
    else:
        print("No image returned. Try starting the prompt with 'Generate an image of...'.")
        if "promptFeedback" in data:
            print(json.dumps(data["promptFeedback"], indent=2))


if __name__ == "__main__":
    if not KEY:
        sys.exit('Set your key first:  export GEMINI_API_KEY="your-key"')
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--models":
        list_models()
    else:
        for p in sys.argv[2:]:
            if not os.path.isfile(p):
                sys.exit(f"File not found: {p}")
        for i in range(1, COUNT + 1):
            print(f"Generating {i}/{COUNT} with {MODEL}...")
            generate(sys.argv[1], sys.argv[2:], i)
