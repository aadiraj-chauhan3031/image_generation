#!/usr/bin/env python3
"""
QA checker for generated jewellery images.

Applies a mm grid to each output image, asks Gemini to measure the bangle
width by reading the grid, then approves or rejects based on ±10% of the
expected size.

Usage:
  python3 qa_check.py output_*.jpeg --expected-mm 60
  python3 qa_check.py output_150600_1.jpeg output_150618_1.jpeg --expected-mm 60

Options:
  --expected-mm   Real-world outer diameter of the product in mm (default: 60)
  --tolerance     Allowed deviation as a fraction (default: 0.10 = 10%)
  --px-per-mm     Scale of the model photos (default: 4.25)
  --save-gridded  Save the gridded versions of output images for visual inspection
"""

import base64, json, os, sys, time, urllib.request, urllib.error, argparse
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Install Pillow first:  pip3 install pillow")

API   = "https://generativelanguage.googleapis.com/v1beta"
KEY   = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3-pro-image"

GRID_COLOR  = (220, 30, 30, 140)
LABEL_COLOR = (220, 30, 30, 230)


def add_grid(img_path, px_per_mm, grid_mm=10):
    img     = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    step    = int(px_per_mm * grid_mm)
    w, h    = img.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    col = 0
    x   = 0
    while x < w:
        lw    = 2 if col % 5 == 0 else 1
        color = (220, 30, 30, 200) if col % 5 == 0 else GRID_COLOR
        draw.line([(x, 0), (x, h)], fill=color, width=lw)
        draw.text((x + 3, 4), f"{col * grid_mm}mm", fill=LABEL_COLOR, font=font)
        x += step; col += 1
    row = 0; y = 0
    while y < h:
        lw    = 2 if row % 5 == 0 else 1
        color = (220, 30, 30, 200) if row % 5 == 0 else GRID_COLOR
        draw.line([(0, y), (w, y)], fill=color, width=lw)
        if row > 0:
            draw.text((4, y + 3), f"{row * grid_mm}mm", fill=LABEL_COLOR, font=font)
        y += step; row += 1
    return Image.alpha_composite(img, overlay).convert("RGB")


def ask_gemini(img: Image.Image, expected_mm: float) -> dict:  # noqa: E501
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        f"This image has a red millimetre grid overlay. "
        f"Look at the cuff bangle on the wrist. "
        f"Using the mm labels on the grid, estimate the outer width of the bangle "
        f"from its leftmost visible edge to its rightmost visible edge. "
        f"The expected width is {expected_mm}mm. "
        f"Reply in this exact format on one line: "
        f"MEASURED=<number> VERDICT=PASS or VERDICT=FAIL NOTE=<one sentence> "
        f"where PASS means within 10% of {expected_mm}mm ({expected_mm*0.9:.1f}–{expected_mm*1.1:.1f}mm)."
    )

    body = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {}
    }

    req = urllib.request.Request(
        f"{API}/models/gemini-3.5-flash:generateContent",
        headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
        data=json.dumps(body).encode()
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        return {"raw": f"API error {e.code}: {e.read().decode()}", "verdict": "ERROR", "measured_mm": None}

    text = ""
    for c in data.get("candidates", []):
        for part in c.get("content", {}).get("parts", []):
            if "text" in part:
                text += part["text"]

    import re
    result = {"raw": text.strip(), "verdict": "ERROR", "measured_mm": None}

    # Try strict format first: MEASURED=<number>
    m = re.search(r"MEASURED=([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        result["measured_mm"] = float(m.group(1))
    else:
        # Fall back: extract any standalone number followed by mm
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
        if nums:
            # Pick the number most likely to be the bangle width (30-120mm range)
            candidates = [float(n) for n in nums if 20 <= float(n) <= 150]
            if candidates:
                result["measured_mm"] = candidates[0]

    # Determine verdict from measured value if present, else from text
    if result["measured_mm"] is not None:
        low  = expected_mm * 0.9
        high = expected_mm * 1.1
        result["verdict"] = "PASS" if low <= result["measured_mm"] <= high else "FAIL"
    elif "VERDICT=PASS" in text:
        result["verdict"] = "PASS"
    elif "VERDICT=FAIL" in text:
        result["verdict"] = "FAIL"

    return result


def main():
    if not KEY:
        sys.exit('Set your key first:  export GEMINI_API_KEY="your-key"')

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images",        nargs="+",          help="output images to check")
    p.add_argument("--expected-mm", type=float, default=60.0)
    p.add_argument("--tolerance",   type=float, default=0.10)
    p.add_argument("--px-per-mm",   type=float, default=4.25)
    p.add_argument("--save-gridded", action="store_true")
    args = p.parse_args()

    low  = args.expected_mm * (1 - args.tolerance)
    high = args.expected_mm * (1 + args.tolerance)

    passed, failed, errors = [], [], []

    print(f"\nQA Check — expected {args.expected_mm}mm ±{args.tolerance*100:.0f}%  ({low:.1f}–{high:.1f}mm)\n")
    print(f"{'Image':<35} {'Measured':>10} {'Verdict':>8}  Notes")
    print("─" * 80)

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"{img_path:<35} {'FILE NOT FOUND':>10}")
            errors.append(img_path)
            continue

        gridded = add_grid(img_path, args.px_per_mm)

        if args.save_gridded:
            name, ext = os.path.splitext(img_path)
            gridded.save(f"qa_{name}{ext}")

        result = ask_gemini(gridded, args.expected_mm)

        mm_str  = f"{result['measured_mm']:.1f}mm" if result["measured_mm"] else "—"
        verdict = result["verdict"]
        note    = result["raw"].split("NOTE=")[-1].strip() if "NOTE=" in result["raw"] else result["raw"][:60]

        icon = "✓" if verdict == "PASS" else ("✗" if verdict == "FAIL" else "?")
        print(f"{img_path:<35} {mm_str:>10} {icon+' '+verdict:>8}  {note}")

        if verdict == "PASS":
            passed.append(img_path)
        elif verdict == "FAIL":
            failed.append(img_path)
        else:
            errors.append(img_path)

        time.sleep(1)

    print("─" * 80)
    print(f"\nSummary: {len(passed)} PASSED  |  {len(failed)} FAILED  |  {len(errors)} ERRORS")

    if passed:
        print(f"\n✓ Approved:  {', '.join(passed)}")
    if failed:
        print(f"✗ Rejected:  {', '.join(failed)}")
    if errors:
        print(f"? Errors:    {', '.join(errors)}")


if __name__ == "__main__":
    main()
