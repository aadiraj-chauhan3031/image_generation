#!/usr/bin/env python3
"""Retry measurement for existing DPBN0002 generated images."""
import os, sys
sys.path.insert(0, "/Users/aadirajchauhan/Desktop/Practice coding/codeVector_work/image_generation")

import gen_compare_dpbn0002 as p

OUT = p.OUT
imgs = sorted(
    [os.path.join(OUT, f) for f in os.listdir(OUT)
     if f.startswith("dpbn0002_") and f.endswith((".jpeg", ".jpg", ".png"))],
    key=lambda x: os.path.basename(x)
)
print(f"Found {len(imgs)} images to process:")
for i in imgs:
    print(f"  {os.path.basename(i)}")

results = []
for path in imgs:
    print(f"\n--- {os.path.basename(path)} ---")
    px_per_mm, c1, c2 = p.estimate_scale(path)
    if px_per_mm is None:
        print("  Could not estimate scale — skipping")
        continue
    print(f"  Scale: {px_per_mm:.2f} px/mm  |  D1 {c1}  D2 {c2}")

    d1_mm, d2_mm = p.measure_with_gemini(path, px_per_mm, c1, c2)
    if d1_mm is None or d2_mm is None:
        print("  Measurement failed")
        continue

    diff1 = abs(d1_mm - p.SPEC_D1_MM) / p.SPEC_D1_MM * 100
    diff2 = abs(d2_mm - p.SPEC_D2_MM) / p.SPEC_D2_MM * 100
    ok = "✓" if diff1 <= 10 and diff2 <= 10 else "✗"
    print(f"  {ok} D1={d1_mm:.2f}mm (Δ{diff1:.0f}%)  D2={d2_mm:.2f}mm (Δ{diff2:.0f}%)")

    comp = p.make_comparison(path, d1_mm, d2_mm, px_per_mm, c1, c2)
    print(f"  Comparison: {comp}")
    results.append({"path": path, "comp": comp, "d1": d1_mm, "d2": d2_mm, "err1": diff1, "err2": diff2})

print("\n=== SUMMARY ===")
for r in results:
    ok = "✓" if r["err1"] <= 10 and r["err2"] <= 10 else "✗"
    print(f"{ok} {os.path.basename(r['path'])}: D1={r['d1']:.2f}mm (Δ{r['err1']:.0f}%)  D2={r['d2']:.2f}mm (Δ{r['err2']:.0f}%)")

if results:
    best = min(results, key=lambda r: r["err1"] + r["err2"])
    print(f"\nBest: {os.path.basename(best['path'])}  → {os.path.basename(best['comp'])}")
