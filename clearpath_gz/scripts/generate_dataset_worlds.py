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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', choices=sorted(gcw.SCENE_PRESETS), required=True)
    p.add_argument('--count', type=int, required=True, help='Total worlds to generate for this scene.')
    p.add_argument('--train', type=int, required=True)
    p.add_argument('--val', type=int, required=True)
    p.add_argument('--test', type=int, required=True)
    p.add_argument('--root-seed', dest='root_seed', type=int, required=True,
                   help='terrain_seed = root_seed + i, object_seed = root_seed + %d + i.' % OBJECT_SEED_OFFSET)
    p.add_argument('--start-index', dest='start_index', type=int, default=0,
                   help='First <scene>_NN index; bump when appending more worlds to a scene.')
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    # Heightmap params (forwarded to generate_heightmap.generate).
    p.add_argument('--octaves', type=int, default=generate_heightmap.DEFAULT_OCTAVES)
    p.add_argument('--persistence', type=float, default=generate_heightmap.DEFAULT_PERSISTENCE)
    p.add_argument('--craters', type=int, default=0)
    p.add_argument('--crater-depth', dest='crater_depth', type=float,
                   default=generate_heightmap.DEFAULT_CRATER_DEPTH)
    p.add_argument('--size', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=[20.0, 20.0, 1.0],
                   help='World extents of the heightmap in meters.')
    # Object placement params (forwarded to generate_custom_world.generate_world).
    p.add_argument('--object-count', dest='object_count', type=int, default=gcw.DEFAULT_OBJECT_COUNT)
    p.add_argument('--object-kind', dest='object_kind', choices=['include', 'mesh'], default='include')
    p.add_argument('--model-uri', dest='model_uris', action='append', default=None)
    p.add_argument('--name-prefix', dest='name_prefix', default=gcw.DEFAULT_NAME_PREFIX)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--scale-jitter', dest='scale_jitter', type=float, default=0.0)
    p.add_argument('--texture', type=Path, default=None)
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
        terrain_seed = args.root_seed + i
        object_seed = args.root_seed + OBJECT_SEED_OFFSET + i

        field = generate_heightmap.generate(
            generate_heightmap.DEFAULT_SIZE, terrain_seed, args.octaves, args.persistence,
            args.craters, args.crater_depth,
        )
        hm_path = HEIGHTMAPS_DIR / f"{world_id}_heightmap.png"
        Image.fromarray(np.round(field * 255).astype(np.uint8), mode='L').save(hm_path)

        sdf, placements = gcw.generate_world(
            hm_path, tuple(args.size), gcw.HEIGHTMAP_POS,
            with_objects=args.object_count > 0, object_count=args.object_count,
            seed=object_seed, model_uris=model_uris, name_prefix=args.name_prefix,
            world_name=world_id, texture_path=texture_path, scene=args.scene,
            object_kind=args.object_kind, scale=args.scale, scale_jitter=args.scale_jitter,
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
            "craters": args.craters,
            "object_count": len(placements),
        }
        print(f"[INFO] [{splits[i]}] {world_id}: heightmap={hm_path.name} world={world_path.name} "
              f"terrain_seed={terrain_seed} object_seed={object_seed} objects={len(placements)}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"[INFO] Wrote manifest: {args.manifest} ({len(manifest['worlds'])} worlds total)")


if __name__ == "__main__":
    main()
