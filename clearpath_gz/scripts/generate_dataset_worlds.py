"""Batch-generate heightmap+world pairs for one scene, with a train/val/test
manifest, for traversability dataset collection.

Calls generate_heightmap.generate() and generate_custom_world.generate_world()
directly (sibling-module imports) rather than shelling out, so heightmap and
world generation share one seeded loop per dataset entry. Run once per scene;
re-running with the same --scene/--start-index upserts those entries in the
manifest instead of duplicating them.

Sample usage:
# Mars: rocks + bare terrain, no craters.
python3 scripts/generate_dataset_worlds.py --scene mars --count 10 \\
    --train 6 --val 2 --test 2 --root-seed 1000 \\
    --object-count 24 --object-kind mesh --model-uri 'model://martian_rock3' --scale 0.8

# Moon: smaller rocks + craters.
python3 scripts/generate_dataset_worlds.py --scene moon --count 10 \\
    --train 6 --val 2 --test 2 --root-seed 2000 --craters 8 \\
    --object-count 16 --object-kind mesh --model-uri 'model://martian_rock3' --scale 0.5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import generate_heightmap
import generate_custom_world as gcw

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
HEIGHTMAPS_DIR = PACKAGE_DIR / 'heightmaps'
WORLDS_DIR = PACKAGE_DIR / 'worlds'
DEFAULT_MANIFEST = WORLDS_DIR / 'dataset_manifest.json'
# object_seed lives in a disjoint range from terrain_seed so the two RNG
# streams (independently seeded per world) never collide across a scene.
OBJECT_SEED_OFFSET = 10_000
# Third disjoint range, for the per-world parameter draws (--*-range). Keeping it
# clear of both other streams means enabling sampling cannot shift the terrain or
# object layout a world would otherwise have had.
PARAM_SEED_OFFSET = 20_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', choices=sorted(gcw.SCENE_PRESETS), required=True)
    p.add_argument('--count', type=int, required=True, help='Total worlds to generate for this scene.')
    p.add_argument('--train', type=int, required=True)
    p.add_argument('--val', type=int, required=True)
    p.add_argument('--test', type=int, required=True)
    p.add_argument('--root-seed', dest='root_seed', type=int, required=True,
                   help='terrain_seed = root_seed + idx, object_seed = root_seed + %d + idx '
                        '(idx = start_index + position).' % OBJECT_SEED_OFFSET)
    p.add_argument('--start-index', dest='start_index', type=int, default=0,
                   help='First <scene>_NN index; bump when appending more worlds to a scene.')
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    # Heightmap params (forwarded to generate_heightmap.generate).
    p.add_argument('--octaves', type=int, default=generate_heightmap.DEFAULT_OCTAVES)
    p.add_argument('--persistence', type=float, default=generate_heightmap.DEFAULT_PERSISTENCE)
    p.add_argument('--craters', type=int, default=0)
    p.add_argument('--crater-depth', dest='crater_depth', type=float,
                   default=generate_heightmap.DEFAULT_CRATER_DEPTH)
    p.add_argument('--crater-radius', dest='crater_radius', type=float, nargs=2,
                   metavar=('MIN', 'MAX'),
                   default=list(generate_heightmap.CRATER_RADIUS_RANGE),
                   help='Crater radius range as a fraction of the heightmap side. '
                        'This is the occlusion-size knob, not --crater-depth.')
    p.add_argument('--size', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=[20.0, 20.0, 1.0],
                   help='World extents of the heightmap in meters.')
    # Object placement params (forwarded to generate_custom_world.generate_world).
    p.add_argument('--object-count', dest='object_count', type=int, default=gcw.DEFAULT_OBJECT_COUNT)
    p.add_argument('--object-kind', dest='object_kind', choices=['include', 'mesh'], default='include')
    p.add_argument('--model-uri', dest='model_uris', action='append', default=None)
    p.add_argument('--name-prefix', dest='name_prefix', default=gcw.DEFAULT_NAME_PREFIX)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--scale-jitter', dest='scale_jitter', type=float, default=0.0)
    p.add_argument('--scale-band', dest='scale_bands', type=float, nargs=3,
                   action='append', metavar=('MIN', 'MAX', 'WEIGHT'),
                   help='Per-rock scale interval and relative probability; repeatable. '
                        'Overrides --scale/--scale-jitter.')
    p.add_argument('--texture', type=Path, default=None)
    # --- per-world parameter sampling -------------------------------------------
    # Without these every world in a batch is identical except for its seeds, so a
    # scene's worlds are all draws from one narrow distribution and the 20th world
    # teaches about as much as the 100th. Each range overrides its scalar counterpart
    # and is sampled once per world from a dedicated RNG stream, then recorded in the
    # manifest so the world stays exactly reproducible and the realised distribution
    # can be audited afterwards.
    g = p.add_argument_group('per-world parameter sampling (override the scalars above)')
    g.add_argument('--object-count-range', dest='object_count_range', type=int, nargs=2,
                   metavar=('LO', 'HI'), default=None)
    g.add_argument('--craters-range', dest='craters_range', type=int, nargs=2,
                   metavar=('LO', 'HI'), default=None)
    g.add_argument('--crater-depth-range', dest='crater_depth_range', type=float, nargs=2,
                   metavar=('LO', 'HI'), default=None)
    g.add_argument('--crater-radius-range', dest='crater_radius_range', type=float, nargs=2,
                   metavar=('LO', 'HI'), default=None,
                   help='Two values are drawn from this range per world and sorted into '
                        'that world\'s (min, max) crater radius, so both crater size and '
                        'size *spread* vary between worlds.')
    g.add_argument('--scale-range', dest='scale_range', type=float, nargs=2,
                   metavar=('LO', 'HI'), default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.train + args.val + args.test != args.count:
        raise SystemExit(
            f"--train/--val/--test ({args.train}+{args.val}+{args.test}) must sum to --count ({args.count})"
        )
    splits = ['train'] * args.train + ['val'] * args.val + ['test'] * args.test

    manifest = {"worlds": {}}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        manifest.setdefault("worlds", {})

    model_uris = args.model_uris if args.model_uris else gcw.DEFAULT_MODEL_URIS
    texture_path = (args.texture or gcw.SOLID_TEXTURE).resolve()
    if not texture_path.exists():
        if texture_path == gcw.SOLID_TEXTURE.resolve():
            gcw.ensure_solid_texture(texture_path)
        else:
            raise SystemExit(f"Texture not found: {texture_path}")

    HEIGHTMAPS_DIR.mkdir(parents=True, exist_ok=True)
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        idx = args.start_index + i
        world_id = f"{args.scene}_{idx:02d}"
        # Seeds key off idx, not i, so appending with --start-index yields fresh
        # terrain instead of silently re-rolling the seeds an earlier run used.
        # Identical to `+ i` for the default --start-index 0, so worlds already
        # generated stay reproducible from their original command.
        terrain_seed = args.root_seed + idx
        object_seed = args.root_seed + OBJECT_SEED_OFFSET + idx
        # Third disjoint stream, so sampling parameters cannot perturb the terrain or
        # object draws a world would otherwise have got.
        param_seed = args.root_seed + PARAM_SEED_OFFSET + idx

        rng = np.random.default_rng(param_seed)
        craters = (int(rng.integers(args.craters_range[0], args.craters_range[1] + 1))
                   if args.craters_range else args.craters)
        crater_depth = (float(rng.uniform(*args.crater_depth_range))
                        if args.crater_depth_range else args.crater_depth)
        crater_radius = (sorted(float(v) for v in rng.uniform(*args.crater_radius_range, size=2))
                         if args.crater_radius_range else list(args.crater_radius))
        object_count = (int(rng.integers(args.object_count_range[0], args.object_count_range[1] + 1))
                        if args.object_count_range else args.object_count)
        scale = (float(rng.uniform(*args.scale_range))
                 if args.scale_range else args.scale)

        field = generate_heightmap.generate(
            generate_heightmap.DEFAULT_SIZE, terrain_seed, args.octaves, args.persistence,
            craters, crater_depth, tuple(crater_radius),
        )
        hm_path = HEIGHTMAPS_DIR / f"{world_id}_heightmap.png"
        Image.fromarray(np.round(field * 255).astype(np.uint8), mode='L').save(hm_path)

        sdf, placements = gcw.generate_world(
            hm_path, tuple(args.size), gcw.HEIGHTMAP_POS,
            with_objects=object_count > 0, object_count=object_count,
            seed=object_seed, model_uris=model_uris, name_prefix=args.name_prefix,
            world_name=world_id, texture_path=texture_path, scene=args.scene,
            object_kind=args.object_kind, scale=scale, scale_jitter=args.scale_jitter,
            scale_bands=args.scale_bands,
        )
        world_path = WORLDS_DIR / f"{world_id}.sdf"
        world_path.write_text(sdf, encoding='utf-8')

        if placements:
            positions_path = WORLDS_DIR / f"{world_id}_objects.json"
            positions_path.write_text(json.dumps({
                "world": world_id,
                "heightmap": str(hm_path.relative_to(PACKAGE_DIR)),
                "size": list(args.size),
                "pos": list(gcw.HEIGHTMAP_POS),
                "seed": object_seed,
                "objects": placements,
            }, indent=2), encoding='utf-8')

        manifest["worlds"][world_id] = {
            "scene": args.scene,
            "split": splits[i],
            "heightmap": str(hm_path.relative_to(PACKAGE_DIR)),
            "world": str(world_path.relative_to(PACKAGE_DIR)),
            "terrain_seed": terrain_seed,
            "object_seed": object_seed,
            "param_seed": param_seed,
            "craters": craters,
            "crater_radius": crater_radius,
            # Recorded even when not sampled: without these the manifest cannot
            # reproduce a world, and the realised parameter spread cannot be audited.
            "crater_depth": crater_depth,
            "scale": scale,
            "scale_bands": args.scale_bands,
            "object_count_requested": object_count,
            "object_count": len(placements),
        }
        print(f"[INFO] [{splits[i]}] {world_id}: heightmap={hm_path.name} world={world_path.name} "
              f"terrain_seed={terrain_seed} object_seed={object_seed} objects={len(placements)} "
              f"craters={craters} radius=({crater_radius[0]:.3f},{crater_radius[1]:.3f}) "
              f"depth={crater_depth:.3f} scale={scale:.2f}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"[INFO] Wrote manifest: {args.manifest} ({len(manifest['worlds'])} worlds total)")


if __name__ == "__main__":
    main()
