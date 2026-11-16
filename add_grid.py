#!/usr/bin/env python3
"""
Overlay a millimetre grid on an image for accurate scale reference.

Requirements:
  pip3 install pillow

Workflow:
  Step 1 — Measure the bangle diameter in pixels on the CAD image:
            Open DPBN0008.jpg in Preview → Tools → Show Inspector → hover over
            the bangle circle and note its width in pixels.
            OR just run with --measure to get an interactive guide.

  Step 2 — Run on the CAD image with the known real-world size:
            python3 add_grid.py DPBN0008.jpg --ref-px 240 --ref-mm 60
            (replace 240 with the pixel diameter you measured)
            It will print the px/mm ratio — copy that number.

  Step 3 — Run on the model image with the same px/mm ratio:
            python3 add_grid.py model_image.jpeg --px-per-mm 4.0
            (replace 4.0 with the number from Step 2)

  Output: gridded_DPBN0008.jpg and gridded_model_image.jpeg

  Step 4 — Pass both gridded images to gen.py as references.
"""

import sys, os, argparse
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow not installed. Run:  pip3 install pillow")

GRID_COLOR   = (220, 30, 30, 140)   # semi-transparent red
LABEL_COLOR  = (220, 30, 30, 230)
LINE_WIDTH   = 1
MAJOR_EVERY  = 5   # thicker line every N cells


def add_grid(img_path, px_per_mm, grid_mm=10):
    img     = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    step = int(px_per_mm * grid_mm)
    w, h = img.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    col = 0
    x   = 0
    while x < w:
        lw    = 2 if col % MAJOR_EVERY == 0 else LINE_WIDTH
        color = (220, 30, 30, 200) if col % MAJOR_EVERY == 0 else GRID_COLOR
        draw.line([(x, 0), (x, h)], fill=color, width=lw)
        draw.text((x + 3, 4), f"{col * grid_mm}mm", fill=LABEL_COLOR, font=font)
        x   += step
        col += 1

    row = 0
    y   = 0
    while y < h:
        lw    = 2 if row % MAJOR_EVERY == 0 else LINE_WIDTH
        color = (220, 30, 30, 200) if row % MAJOR_EVERY == 0 else GRID_COLOR
        draw.line([(0, y), (w, y)], fill=color, width=lw)
        if row > 0:
            draw.text((4, y + 3), f"{row * grid_mm}mm", fill=LABEL_COLOR, font=font)
        y   += step
        row += 1

    result = Image.alpha_composite(img, overlay).convert("RGB")
    name, ext = os.path.splitext(os.path.basename(img_path))
    out = f"gridded_{name}{ext}"
    result.save(out)
    print(f"Saved: {out}  |  scale: {px_per_mm:.4f} px/mm  |  grid: {grid_mm}mm cells")
    return px_per_mm


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image",                              help="image file to add grid to")
    p.add_argument("--ref-px",   type=float,             help="pixel span of a known feature in this image")
    p.add_argument("--ref-mm",   type=float,             help="real-world mm size of that feature")
    p.add_argument("--px-per-mm", type=float,            help="pixels per mm (reuse ratio from first image)")
    p.add_argument("--grid-mm",  type=float, default=10, help="grid cell size in mm (default: 10)")
    args = p.parse_args()

    if args.px_per_mm:
        ratio = args.px_per_mm
    elif args.ref_px and args.ref_mm:
        ratio = args.ref_px / args.ref_mm
        print(f"\nScale calculated: {ratio:.4f} px/mm")
        print(f"Use this for other images:  --px-per-mm {ratio:.4f}\n")
    else:
        p.error("Provide either --px-per-mm OR both --ref-px and --ref-mm.\n\n"
                "To measure the bangle diameter in pixels:\n"
                "  Open the image in Preview → Tools → Annotate → Measure\n"
                "  Draw a line across the bangle diameter and note the pixel count.")

    add_grid(args.image, ratio, args.grid_mm)
