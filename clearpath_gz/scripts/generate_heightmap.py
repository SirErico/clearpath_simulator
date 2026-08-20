r"""Procedurally generate a greyscale terrain heightmap PNG.

Reuses the fractal-noise field from generate_surface_texture.py (same
octave-sum technique, just written out as elevation instead of colorized)
so terrain "roughness" and texture "look" can optionally share a seed.

Sample usage:
# Rolling hills, no craters:
python3 scripts/generate_heightmap.py --seed 3 --output heightmaps/heightmap_03.png
# Cratered moon-like terrain:
python3 scripts/generate_heightmap.py --seed 5 --craters 10 --crater-depth 0.2 \    
    --output heightmaps/moon_heightmap_05.png
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from generate_surface_texture import fractal_noise

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
HEIGHTMAPS_DIR = PACKAGE_DIR / 'heightmaps'

DEFAULT_SIZE = 513  # 2^n + 1, required by Gazebo heightmap geometry
# Low octaves + low persistence => broad smooth swells, not small bumps.
# The intent is that relief comes from placed rocks/craters, not the base
# terrain noise itself.
DEFAULT_OCTAVES = 4
DEFAULT_PERSISTENCE = 0.35
# Default radius range as a fraction of the image side; --crater-radius overrides.
# Radius, not depth, is what governs how much terrain a crater self-occludes.
CRATER_RADIUS_RANGE = (0.02, 0.08)  # fraction of image side
CRATER_RIM_HEIGHT = 0.05
CRATER_RIM_WIDTH = 0.35  # fraction of radius; controls rim ring falloff
DEFAULT_CRATER_DEPTH = 0.18


def _is_pow2_plus_one(n: int) -> bool:
    m = n - 1
    return m > 0 and (m & (m - 1)) == 0


def crater_field(shape: tuple[int, int], count: int, radius_px_range: tuple[float, float],
                 depth: float, rng: np.random.Generator) -> np.ndarray:
    """Additive elevation delta: a paraboloid bowl (negative) inside each
    crater plus a raised rim ring right at its edge. Overlapping craters
    stack additively, which reads fine for scattered impact craters."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(count):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        radius = rng.uniform(*radius_px_range)
        rn = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
        bowl = np.where(rn <= 1.0, depth * (rn ** 2 - 1.0), 0.0)
        rim = CRATER_RIM_HEIGHT * np.exp(-((rn - 1.0) / CRATER_RIM_WIDTH) ** 2)
        field += bowl + rim
    return field


def generate(size: int, seed: int, octaves: int, persistence: float,
            craters: int, crater_depth: float,
            radius_range: tuple[float, float] = CRATER_RADIUS_RANGE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (size, size)
    field = fractal_noise(shape, octaves, rng, persistence=persistence)
    if craters > 0:
        radius_px = (radius_range[0] * size, radius_range[1] * size)
        field = field + crater_field(shape, craters, radius_px, crater_depth, rng)
        # Clip (not renormalize) so crater depth stays consistent across seeds —
        # renormalizing would rescale the whole field by whatever the deepest
        # crater happened to be that run.
        field = np.clip(field, 0.0, 1.0)
    return field


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    p.add_argument('--size', type=int, default=DEFAULT_SIZE,
                   help='Square heightmap side in pixels. Gazebo expects 2^n + 1.')
    p.add_argument('--output', type=Path, default=None,
                   help='Output PNG path. Default: heightmaps/heightmap_<seed>.png')
    p.add_argument('--octaves', type=int, default=DEFAULT_OCTAVES,
                   help='Octaves for the base fractal-noise terrain.')
    p.add_argument('--persistence', type=float, default=DEFAULT_PERSISTENCE,
                   help='Per-octave amplitude decay. Lower = smoother (fewer small bumps).')
    p.add_argument('--craters', type=int, default=0, help='Number of craters to stamp (0 = none).')
    p.add_argument('--crater-radius', dest='crater_radius', type=float, nargs=2,
                   metavar=('MIN', 'MAX'), default=list(CRATER_RADIUS_RANGE),
                   help='Crater radius range as a fraction of the image side '
                        f'(default {CRATER_RADIUS_RANGE[0]} {CRATER_RADIUS_RANGE[1]}). '
                        'This is the occlusion-size knob, not --crater-depth.')
    p.add_argument('--crater-depth', dest='crater_depth', type=float, default=DEFAULT_CRATER_DEPTH,
                   help='Crater bowl depth (fraction of the [0,1] elevation range).')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _is_pow2_plus_one(args.size):
        print(f"[WARN] --size {args.size} is not 2^n+1; Gazebo expects e.g. 129, 257, 513.")

    if args.seed is None:
        print("[WARN] No --seed given; the saved command will not reproduce this heightmap.")

    output = args.output
    if output is None:
        stem = f"heightmap_{args.seed:02d}" if args.seed is not None else "heightmap"
        output = HEIGHTMAPS_DIR / f"{stem}.png"

    field = generate(args.size, args.seed, args.octaves, args.persistence,
                     args.craters, args.crater_depth, tuple(args.crater_radius))

    output.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.round(field * 255).astype(np.uint8), mode='L')
    img.save(output)

    command = shlex.join(['python3', sys.argv[0], *sys.argv[1:]])
    meta_path = output.with_name(f"{output.stem}_meta.json")
    meta_path.write_text(json.dumps({
        "command": command,
        "seed": args.seed,
        "size": args.size,
        "octaves": args.octaves,
        "persistence": args.persistence,
        "craters": args.craters,
        "crater_depth": args.crater_depth,
    }, indent=2), encoding='utf-8')

    print(f"[INFO] Generated: {output} ({args.size}x{args.size}, seed {args.seed}, "
          f"craters={args.craters})")
    print(f"[INFO] Wrote metadata: {meta_path}")
    print(f"[INFO] Regenerate with: {command}")


if __name__ == "__main__":
    main()
