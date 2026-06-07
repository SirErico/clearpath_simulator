"""Procedurally generate a multi-tone planetary ground texture PNG.
Sample:
python3 scripts/generate_surface_texture.py --palette moon --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

# Regolith palettes as (stop, (r, g, b)) control points in [0, 1].
PALETTES = {
    "mars": [
        (0.00, (90, 50, 38)),     # basalt shadow
        (0.25, (135, 75, 55)),    # dark rust
        (0.50, (180, 105, 70)),    # mid rust
        (0.70, (210, 135, 88)),   # ochre / orange
        (0.86, (230, 170, 125)),  # light tan
        (1.00, (248, 210, 172)),  # pale dust highlight
    ],
    "moon": [
        # Neutral-to-faintly-cool greys (B >= R) and a lifted floor so the
        # surface reads as bright lunar regolith, not warm/brownish.
        (0.00, (74, 75, 77)),     # deep shadow
        (0.25, (116, 117, 120)),  # dark mare basalt
        (0.50, (152, 153, 156)),  # mid grey
        (0.70, (188, 189, 192)),  # light grey
        (0.86, (216, 217, 220)),  # highland grey
        (1.00, (240, 241, 244)),  # bright highlight
    ],
}
DEFAULT_OUTPUTS = {
    "mars": PACKAGE_DIR / 'textures' / 'mars_texture.png',
    "moon": PACKAGE_DIR / 'textures' / 'moon_texture.png',
}


def fractal_noise(shape: tuple[int, int], octaves: int, rng: np.random.Generator) -> np.ndarray:
    """Sum of bicubically-upsampled random octaves, normalized to [0, 1]."""
    h, w = shape
    field = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    base = 4  # coarsest grid side
    for o in range(octaves):
        side = base * (2 ** o)
        coarse = rng.random((side, side), dtype=np.float32)
        up = np.asarray(
            Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        field += amplitude * up
        total_amp += amplitude
        amplitude *= 0.5
    field /= total_amp
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return field


def apply_palette(t: np.ndarray, palette: list) -> np.ndarray:
    """Map t in [0, 1] (HxW) to an RGB image (HxWx3, float) via palette."""
    stops = np.array([s for s, _ in palette], dtype=np.float32)
    colors = np.array([c for _, c in palette], dtype=np.float32)
    rgb = np.empty((*t.shape, 3), dtype=np.float32)
    for ch in range(3):
        rgb[..., ch] = np.interp(t, stops, colors[:, ch])
    return rgb


def generate(size: int, seed: int, palette: list) -> Image.Image:
    rng = np.random.default_rng(seed)
    shape = (size, size)

    # Base tone from fractal noise gives broad regolith variation.
    base = fractal_noise(shape, octaves=6, rng=rng)

    # Low-frequency blotch field darkens scattered patches.
    blotch = fractal_noise(shape, octaves=3, rng=rng)
    base = base - 0.35 * np.clip(blotch - 0.55, 0.0, None) * 2.0
    base = np.clip(base, 0.0, 1.0)

    rgb = apply_palette(base, palette)

    # Fine grain so the surface is not flat under close-up lidar/cameras.
    grain = rng.normal(0.0, 6.0, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + grain, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb, mode='RGB')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--palette', choices=sorted(PALETTES), default='mars',
                   help='Regolith color palette.')
    p.add_argument('--output', type=Path, default=None,
                   help='Output PNG path (default: textures/<palette>_texture.png).')
    p.add_argument('--size', type=int, default=1024, help='Square texture side in pixels.')
    p.add_argument('--seed', type=int, default=7, help='Random seed for reproducibility.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output is not None else DEFAULT_OUTPUTS[args.palette]
    img = generate(args.size, args.seed, PALETTES[args.palette])
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    print(f"[INFO] Wrote {args.palette} texture: {output} "
          f"({args.size}x{args.size}, seed {args.seed})")


if __name__ == "__main__":
    main()
