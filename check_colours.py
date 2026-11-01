"""Check a chart palette for colour-blind safety, the same way every time.

A Python port of the categorical checks in the data-viz validator (validate_palette.js):
  - lightness band   OKLCH L inside the mode's band
  - chroma floor     OKLCH C >= 0.10 (below it a colour reads as grey)
  - CVD separation   OKLab distance x100 between neighbouring colours under simulated
                     protan/deutan vision (Machado 2009, severity 1): >= 8 passes, 6-8 is
                     allowed only with labels, below 6 fails
  - normal vision    the same distance with normal vision must be >= 15
  - contrast         WCAG ratio against the chart background; under 3:1 needs labels

Run:  python check_colours.py          (checks the app's palette in light and dark)
      python check_colours.py "#2a78d6,#eda100" light "#ffffff"
"""
import sys

import numpy as np

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 0.10, 8.0, 6.0, 15.0, 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
}


def lin(hex_colour):
    s = np.array([int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def oklab(rgb_lin):
    r, g, b = rgb_lin
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def delta_e(a, b, kind=None):
    la, lb = lin(a), lin(b)
    if kind:
        la, lb = (np.clip(np.array(MACHADO[kind]) @ x, 0, 1) for x in (la, lb))
    return 100 * float(np.linalg.norm(oklab(la) - oklab(lb)))


def contrast(a, b):
    lum = lambda h: float(np.dot([0.2126, 0.7152, 0.0722], lin(h)))
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def validate(palette, mode, surface):
    """List of (check, result, detail); result is PASS, WARN or FAIL."""
    lo, hi = BAND[mode]
    lab = {c: oklab(lin(c)) for c in palette}
    pairs = list(zip(palette, palette[1:]))
    out = []
    off = [(c, round(lab[c][0], 3)) for c in palette if not lo <= lab[c][0] <= hi]
    out.append(("Lightness band", "FAIL" if off else "PASS", off or f"all inside L {lo}-{hi}"))
    grey = [(c, round(float(np.hypot(*lab[c][1:])), 3)) for c in palette if np.hypot(*lab[c][1:]) < CHROMA_FLOOR]
    out.append(("Chroma floor", "FAIL" if grey else "PASS", grey or f"all >= {CHROMA_FLOOR}"))
    if pairs:
        cvd = min((delta_e(a, b, k), k, a, b) for a, b in pairs for k in MACHADO)
        state = "PASS" if cvd[0] >= CVD_TARGET else "WARN" if cvd[0] >= CVD_FLOOR else "FAIL"
        out.append(("CVD separation", state, f"worst {cvd[2]}-{cvd[3]} dE {cvd[0]:.1f} ({cvd[1]})"))
        nv = min((delta_e(a, b), a, b) for a, b in pairs)
        out.append(("Normal-vision floor", "PASS" if nv[0] >= NORMAL_FLOOR else "FAIL",
                    f"worst {nv[1]}-{nv[2]} dE {nv[0]:.1f}"))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    out.append(("Contrast vs surface", "WARN" if low else "PASS", low or f"all >= {CONTRAST_MIN}:1"))
    return out


def report(palette, mode, surface):
    print(f"\n{', '.join(palette)}  ({mode}, background {surface})")
    results = validate(palette, mode, surface)
    for name, state, detail in results:
        print(f"  [{state}] {name:<20} {detail}")
    return all(state != "FAIL" for _, state, _ in results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ok = report([c.strip() for c in sys.argv[1].split(",")], sys.argv[2], sys.argv[3])
    else:
        import app_style as s
        ok = all([report([s.SERIES["light"]["green"], s.SERIES["light"]["yellow"]], "light", s.SURFACE["light"]),
                  report([s.SERIES["dark"]["green"], s.SERIES["dark"]["yellow"]], "dark", s.SURFACE["dark"])])
    sys.exit(0 if ok else 1)
