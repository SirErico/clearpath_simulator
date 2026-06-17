"""Generate a Gazebo SDF world from a greyscale heightmap.

The heightmap is used both as the Gazebo terrain (SDF <heightmap> geometry)
and as the elevation source for object placement. World coordinates:
  - heightmap is centered at HEIGHTMAP_POS
  - X spans [pos_x - size_x/2, pos_x + size_x/2]
  - Y spans [pos_y - size_y/2, pos_y + size_y/2]
  - Z = (pixel/255) * size_z + pos_z

The same pipeline drives every scene; --scene only swaps lighting/look and
the objects come from --model-uri, so forest/mars/moon are all one code path.

Sample usage:
# Forest world with the default tree model:
python generate_custom_world.py --output worlds/forest_world.sdf
# Mars world scattered with 32 rocks seated on the surface:
python generate_custom_world.py --scene mars --world-name mars_world \\
    --texture textures/mars_texture.png --object-kind mesh --name-prefix rock \\
    --model-uri 'model://martian_rock3' --object-count 32 --scale 0.8 --scale-jitter 0.3 \\
    --output worlds/mars_world.sdf
"""
import argparse
import itertools
import json
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

HEIGHTMAPS_DIR = PACKAGE_DIR / 'heightmaps'
DEFAULT_OUTPUT = PACKAGE_DIR / 'worlds' / 'forest_world.sdf'
# Default heightmap diffuse texture (overridable with --texture).
SOLID_TEXTURE = PACKAGE_DIR / 'textures' / 'forest_texture.png'

# Default object scattered over the terrain. Pass --model-uri (repeatable) to
# override; with several URIs one is chosen at random per placement.
DEFAULT_MODEL_URI = "model://Pine Tree 2"
DEFAULT_MODEL_URIS = [DEFAULT_MODEL_URI]
DEFAULT_NAME_PREFIX = "tree"
DEFAULT_WORLD_NAME = "forest_world"
DEFAULT_SCENE = "forest"

# Per-planet scene/lighting presets. Keeps the heightmap+spawn pipeline shared
# while letting mars/moon worlds keep their distinct look.
CONTACT_PLUGIN = (
    '    <plugin filename="gz-sim-contact-system" '
    'name="gz::sim::systems::Contact"></plugin>\n'
)
SCENE_PRESETS = {
    "forest": {
        "heightmap": HEIGHTMAPS_DIR / 'forest_heightmap.png',
        "texture_size": 4.0,    # tiles ~5x across a 20 m terrain (grass detail)
        "extra_plugins": "",
        "spherical": True,
        "sun_pose": "0 0 10 0 0 0",
        "sun_intensity": None,
        "sun_direction": "-0.5 0.5 -1.0",
        "ambient": "1 1 1 1",
        "background": "0.3 0.7 0.9 1",
        "grid": None,
    },
    "mars": {
        "heightmap": HEIGHTMAPS_DIR / 'mars_heightmap.png',
        "texture_size": 20.0,   # one full copy across the 20 m terrain
        "extra_plugins": CONTACT_PLUGIN,
        "spherical": False,
        "sun_pose": "0 0 100 0 0 0",
        "sun_intensity": 4,
        "sun_direction": "0.25 -0.5 -0.4",
        "ambient": "0.6 0.52 0.45 1",
        "background": "0.7 0.5 0.3 1",
        "grid": "false",
    },
    "moon": {
        "heightmap": HEIGHTMAPS_DIR / 'moon_heightmap.png',
        "texture_size": 20.0,   # one full copy across the 20 m terrain
        "extra_plugins": CONTACT_PLUGIN,
        "spherical": False,
        "sun_pose": "0 0 100 0 0 0",
        "sun_intensity": 3,
        "sun_direction": "0.25 -0.5 -0.4",
        "ambient": "0.5 0.5 0.5 1",
        "background": "0.05 0.05 0.05 1",
        "grid": "false",
    },
}

# Heightmap world placement (meters). Centered at origin by default.
HEIGHTMAP_POS = (0.0, 0.0, 0.0)

# Object placement defaults
DEFAULT_OBJECT_COUNT = 16
EDGE_MARGIN = 0.5
MAX_ATTEMPTS_FACTOR = 20
GROUND_PENETRATION = 0.08
# Skip objects where local slope exceeds this (rise/run, dimensionless).
MAX_SLOPE = 0.5

# Robot spawn — must match jackal_exploration/launch/sim.launch.py
ROBOT_SPAWN_X = -7.0
ROBOT_SPAWN_Y = -7.0
ROBOT_SPAWN_YAW = np.pi
ROBOT_CLEARANCE_RADIUS = 3.0


def ensure_solid_texture(path: Path, rgb=(80, 130, 70)):
    """Create a small solid-color PNG if it doesn't already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (16, 16), rgb).save(path)
    print(f"[INFO] Created solid texture: {path}")


def load_heightmap(path: Path) -> np.ndarray:
    """Load a greyscale heightmap as a float32 array in [0, 1]."""
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    h, w = arr.shape
    if w != h or (w - 1) & (w - 2):
        # Gazebo requires square images with side = 2^n + 1.
        print(f"[WARN] Heightmap is {w}x{h}; Gazebo expects square 2^n+1 (e.g. 129, 257, 513).")
    return arr


def world_to_image_uv(x: float, y: float, img_shape, size, pos):
    """Map world (x, y) to fractional image (u, v) coords.

    Image convention: (0,0) is top-left; +u is +X world, +v is -Y world.
    """
    h, w = img_shape
    sx, sy, _ = size
    px, py, _ = pos
    u = ((x - px) / sx + 0.5) * (w - 1)
    v = (0.5 - (y - py) / sy) * (h - 1)
    return u, v


def sample_elevation(heightmap: np.ndarray, x: float, y: float, size, pos) -> float:
    """Bilinearly sample world-frame elevation at (x, y)."""
    h, w = heightmap.shape
    u, v = world_to_image_uv(x, y, heightmap.shape, size, pos)
    if u < 0 or u > w - 1 or v < 0 or v > h - 1:
        return pos[2]  # outside heightmap — return base z
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
    du, dv = u - u0, v - v0
    z00 = heightmap[v0, u0]
    z10 = heightmap[v0, u1]
    z01 = heightmap[v1, u0]
    z11 = heightmap[v1, u1]
    z = (z00 * (1 - du) * (1 - dv) + z10 * du * (1 - dv)
         + z01 * (1 - du) * dv + z11 * du * dv)
    return z * size[2] + pos[2]


def local_slope(heightmap: np.ndarray, x: float, y: float, size, pos) -> float:
    """Approximate slope (rise/run) at (x, y) using a small finite difference."""
    sx, sy, _ = size
    h, w = heightmap.shape
    dx = sx / (w - 1)
    dy = sy / (h - 1)
    z_xp = sample_elevation(heightmap, x + dx, y, size, pos)
    z_xm = sample_elevation(heightmap, x - dx, y, size, pos)
    z_yp = sample_elevation(heightmap, x, y + dy, size, pos)
    z_ym = sample_elevation(heightmap, x, y - dy, size, pos)
    gx = (z_xp - z_xm) / (2 * dx)
    gy = (z_yp - z_ym) / (2 * dy)
    return float(np.hypot(gx, gy))


# --- Mesh-object support (rocks etc.) ---------------------------------------
# Rock GLBs are modelled Y-up, so their model.sdf tilts the link +90 deg about
# X to make them Z-up. We replicate that tilt and read the GLB bounds so each
# rock can be scaled and seated on the surface instead of buried.
DEFAULT_MODELS_ROOT = PACKAGE_DIR / 'models'
DEFAULT_MESH_RPY = (np.pi / 2, 0.0, 0.0)


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """SDF roll-pitch-yaw -> rotation matrix (Rz @ Ry @ Rx)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def model_name_from_uri(model_uri: str) -> str:
    """model://<name>[/...] -> <name>."""
    return model_uri.split('//', 1)[-1].split('/', 1)[0]


def resolve_glb(model_uri: str, models_root: Path) -> Path:
    """model://<name> -> <models_root>/**/<name>/meshes/<name>.glb."""
    name = model_name_from_uri(model_uri)
    matches = sorted(models_root.rglob(f"{name}/meshes/{name}.glb"))
    if not matches:
        raise SystemExit(f"Could not locate GLB for {model_uri} under {models_root}")
    return matches[0]


def mesh_world_z_range(glb_path: Path, rpy: tuple[float, float, float]) -> tuple[float, float]:
    """(zmin, zmax) of the unit-scale mesh after applying the link rpy tilt."""
    import trimesh  # optional dep; only needed for --object-kind mesh
    mesh = trimesh.load(glb_path, force='mesh')
    lo, hi = mesh.bounds
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    z = (corners @ _rpy_matrix(*rpy).T)[:, 2]
    return float(z.min()), float(z.max())


def generate_object_block(heightmap: np.ndarray, size, pos, object_count: int, seed=None,
                          model_uris=DEFAULT_MODEL_URIS, name_prefix=DEFAULT_NAME_PREFIX,
                          ground_penetration=GROUND_PENETRATION, max_slope=MAX_SLOPE,
                          object_kind="include", scale=1.0, scale_jitter=0.0, embed_frac=0.1,
                          models_root=DEFAULT_MODELS_ROOT, mesh_rpy=DEFAULT_MESH_RPY):
    """Place objects at random valid points sampled from the heightmap footprint.

    object_kind="include" spawns each model as a plain <include> (base-origin
    objects like trees). object_kind="mesh" spawns an inline <model> with a
    scaled mesh and seats it on the surface using the GLB bounds — needed for
    center-origin rocks and for any per-instance --scale.

    Returns (sdf_blocks, placements) where placements records each object's pose.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    placements = []
    skipped_spawn = 0
    skipped_slope = 0
    attempts = 0
    max_attempts = max(1, object_count * MAX_ATTEMPTS_FACTOR)
    clearance_sq = ROBOT_CLEARANCE_RADIUS ** 2
    cx, cy = pos[0], pos[1]
    half_x = size[0] / 2
    half_y = size[1] / 2

    min_x = cx - half_x + EDGE_MARGIN
    max_x = cx + half_x - EDGE_MARGIN
    min_y = cy - half_y + EDGE_MARGIN
    max_y = cy + half_y - EDGE_MARGIN

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Heightmap area too small after applying EDGE_MARGIN.")

    # Cache unit-scale mesh bounds and resolved GLB path per model.
    mesh_cache = {}
    if object_kind == "mesh":
        for uri in set(model_uris):
            glb = resolve_glb(uri, models_root)
            mesh_cache[uri] = (glb, mesh_world_z_range(glb, mesh_rpy))

    rx, ry, rz = mesh_rpy
    while len(blocks) < object_count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)

        if (x - ROBOT_SPAWN_X) ** 2 + (y - ROBOT_SPAWN_Y) ** 2 < clearance_sq:
            skipped_spawn += 1
            continue

        if local_slope(heightmap, x, y, size, pos) > max_slope:
            skipped_slope += 1
            continue

        ground = sample_elevation(heightmap, x, y, size, pos)
        yaw = rng.uniform(-np.pi, np.pi)
        obj_id = len(blocks)
        model_uri = model_uris[rng.integers(len(model_uris))]
        name = f"{name_prefix}_{obj_id}"
        record = {"name": name, "model": model_uri,
                  "x": round(float(x), 4), "y": round(float(y), 4), "yaw": round(float(yaw), 4)}

        if object_kind == "mesh":
            s = scale * (1.0 + rng.uniform(-scale_jitter, scale_jitter)) if scale_jitter else scale
            glb, (zmin, zmax) = mesh_cache[model_uri]
            # Embed a fraction of the rock's own (scaled) height so small rocks
            # are not swallowed by a fixed offset. Seat the rest above ground.
            embed = embed_frac * s * (zmax - zmin)
            z = ground - embed - s * zmin
            mesh_uri = f"model://{model_name_from_uri(model_uri)}/meshes/{glb.name}"
            blocks.append(f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <pose>0 0 0 {rx:.6f} {ry:.6f} {rz:.6f}</pose>
        <visual name="visual">
          <geometry><mesh><uri>{mesh_uri}</uri><scale>{s:.3f} {2*s:.3f} {s:.3f}</scale></mesh></geometry>
        </visual>
        <collision name="collision">
          <geometry><mesh><uri>{mesh_uri}</uri><scale>{s:.3f} {2*s:.3f} {s:.3f}</scale></mesh></geometry>
        </collision>
      </link>
    </model>""")
            record["z"] = round(float(z), 4)
            record["scale"] = round(float(s), 4)
        else:
            z = ground - ground_penetration
            blocks.append(f"""
    <include>
      <uri>{model_uri}</uri>
      <name>{name}</name>
      <pose>{x:.2f} {y:.2f} {z:.3f} 0 0 {yaw:.3f}</pose>
    </include>""")
            record["z"] = round(float(z), 4)

        placements.append(record)

    print(f"[INFO] Objects placed: {len(blocks)}/{object_count} | attempts: {attempts}/{max_attempts}")
    print(f"[INFO] Objects skipped | spawn: {skipped_spawn} | slope: {skipped_slope}")
    return "\n".join(blocks), placements


def generate_terrain(heightmap_path: Path, size, pos, texture_path: Path = SOLID_TEXTURE,
                     texture_size: float = 1000.0):
    sx, sy, sz = size
    px, py, pz = pos
    return f"""
    <model name='terrain'>
      <static>true</static>
      <pose>{px:.2f} {py:.2f} {pz:.2f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <heightmap>
              <uri>file://{heightmap_path}</uri>
              <size>{sx:.2f} {sy:.2f} {sz:.2f}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.5</mu>
                <mu2>1.5</mu2>
              </ode>
            </friction>
            <contact>
              <ode>
                <kp>1e6</kp>
                <kd>1e2</kd>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name='visual'>
          <geometry>
            <heightmap>
              <uri>file://{heightmap_path}</uri>
              <size>{sx:.2f} {sy:.2f} {sz:.2f}</size>
              <pos>0 0 0</pos>
              <sampling>1</sampling>
              <texture>
                <size>{texture_size:g}</size>
                <diffuse>file://{texture_path}</diffuse>
                <normal>file://{texture_path}</normal>
              </texture>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>"""


def generate_world(heightmap_path: Path, size, pos, with_objects: bool, object_count: int, seed=None,
                   model_uris=DEFAULT_MODEL_URIS, name_prefix=DEFAULT_NAME_PREFIX,
                   world_name=DEFAULT_WORLD_NAME, texture_path: Path = SOLID_TEXTURE,
                   scene=DEFAULT_SCENE, object_kind="include", scale=1.0, scale_jitter=0.0,
                   embed_frac=0.1, texture_size=None):
    """Build the SDF string. Returns (sdf, placements)."""
    preset = SCENE_PRESETS[scene]
    if texture_size is None:
        texture_size = preset["texture_size"]
    heightmap = load_heightmap(heightmap_path)
    terrain_block = generate_terrain(heightmap_path, size, pos, texture_path, texture_size)
    if with_objects:
        object_block, placements = generate_object_block(
            heightmap, size, pos, object_count, seed=seed,
            model_uris=model_uris, name_prefix=name_prefix,
            object_kind=object_kind, scale=scale, scale_jitter=scale_jitter,
            embed_frac=embed_frac)
    else:
        object_block, placements = "    <!-- objects disabled -->", []

    spherical_block = """
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>39.512152</latitude_deg>
      <longitude_deg>22.426669</longitude_deg>
      <elevation>344</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
""" if preset["spherical"] else ""

    intensity_block = (f"      <intensity>{preset['sun_intensity']}</intensity>\n"
                       if preset["sun_intensity"] is not None else "")
    grid_block = f"      <grid>{preset['grid']}</grid>\n" if preset["grid"] is not None else ""

    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{world_name}">

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
{preset["extra_plugins"]}    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <physics name="default_physics" type="ode">
      <max_step_size>0.003</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
{spherical_block}
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>{preset["sun_pose"]}</pose>
      <diffuse>1.0 1.0 1.0 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
{intensity_block}      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>{preset["sun_direction"]}</direction>
    </light>

    <scene>
      <ambient>{preset["ambient"]}</ambient>
      <background>{preset["background"]}</background>
{grid_block}      <shadows>1</shadows>
    </scene>

    {terrain_block}

    <!-- Scattered objects ({name_prefix}) -->
    {object_block}

  </world>
</sdf>
"""
    return sdf, placements


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heightmap', type=Path, default=None,
                   help='Path to greyscale heightmap PNG (square, side = 2^n + 1). '
                        "Defaults to the --scene preset's heightmap_<scene>.png.")
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                   help='Output SDF path.')
    p.add_argument('--size', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   default=[20.0, 20.0, 1.0],
                   help='World extents of the heightmap in meters.')
    p.add_argument('--object-count', dest='object_count', type=int,
                   default=DEFAULT_OBJECT_COUNT,
                   help='Number of objects to attempt to place randomly.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducible object placement.')
    p.add_argument('--no-objects', dest='no_objects', action='store_true',
                   help='Skip object placement (terrain-only world for testing).')
    p.add_argument('--model-uri', dest='model_uris', action='append', default=None,
                   help='Model URI to scatter (repeatable; random pick per placement). '
                        f'Default: {DEFAULT_MODEL_URI}')
    p.add_argument('--name-prefix', default=DEFAULT_NAME_PREFIX,
                   help='Name prefix for placed objects, e.g. "rock".')
    p.add_argument('--world-name', default=DEFAULT_WORLD_NAME,
                   help='SDF <world name>; must match the launch "world" arg and filename stem.')
    p.add_argument('--texture', type=Path, default=SOLID_TEXTURE,
                   help='Heightmap diffuse texture PNG.')
    p.add_argument('--texture-size', dest='texture_size', type=float, default=None,
                   help='Meters spanned by one copy of the texture (smaller = more visible '
                        "tiling/detail). Defaults to the --scene preset's texture_size.")
    p.add_argument('--scene', choices=sorted(SCENE_PRESETS), default=DEFAULT_SCENE,
                   help='Lighting/scene preset.')
    p.add_argument('--object-kind', choices=['include', 'mesh'], default='include',
                   help='"include": plain model include (base-origin, e.g. trees). '
                        '"mesh": inline scaled mesh seated on the surface (center-origin rocks).')
    p.add_argument('--scale', type=float, default=1.0,
                   help='Uniform mesh scale for --object-kind mesh (e.g. 0.5 = half size).')
    p.add_argument('--scale-jitter', type=float, default=0.0,
                   help='Random +/- fraction on scale per object (e.g. 0.3 -> x0.7..x1.3).')
    p.add_argument('--embed-frac', type=float, default=0.1,
                   help='Fraction of each mesh object height buried below the surface.')
    p.add_argument('--no-positions-file', action='store_true',
                   help='Do not write the <output>_objects.json sidecar of placed poses.')
    return p.parse_args()


def main():
    args = parse_args()
    heightmap = args.heightmap or SCENE_PRESETS[args.scene]["heightmap"]
    heightmap_path = heightmap.resolve()
    if not heightmap_path.exists():
        raise SystemExit(f"Heightmap not found: {heightmap_path}")
    texture_path = args.texture.resolve()
    if not texture_path.exists():
        if texture_path == SOLID_TEXTURE.resolve():
            ensure_solid_texture(texture_path)  # forest default: synthesize a solid color
        else:
            raise SystemExit(
                f"Texture not found: {texture_path}\n"
                f"Generate one with generate_surface_texture.py or pass an existing --texture.")

    if args.object_count < 0:
        raise SystemExit("--object-count must be >= 0")

    model_uris = args.model_uris if args.model_uris else DEFAULT_MODEL_URIS

    sdf, placements = generate_world(heightmap_path, tuple(args.size), HEIGHTMAP_POS,
                                     with_objects=not args.no_objects,
                                     object_count=args.object_count,
                                     seed=args.seed,
                                     model_uris=model_uris,
                                     name_prefix=args.name_prefix,
                                     world_name=args.world_name,
                                     texture_path=texture_path,
                                     scene=args.scene,
                                     object_kind=args.object_kind,
                                     scale=args.scale,
                                     scale_jitter=args.scale_jitter,
                                     embed_frac=args.embed_frac,
                                     texture_size=args.texture_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sdf, encoding='utf-8')

    if not args.no_positions_file and placements:
        positions_path = args.output.with_name(f"{args.output.stem}_objects.json")
        # Store the heightmap relative to the package so the sidecar stays
        # portable across checkouts; fall back to absolute if it lives elsewhere.
        try:
            heightmap_field = str(heightmap_path.relative_to(PACKAGE_DIR))
        except ValueError:
            heightmap_field = str(heightmap_path)
        positions_path.write_text(json.dumps({
            "world": args.world_name,
            "heightmap": heightmap_field,
            "size": list(args.size),
            "pos": list(HEIGHTMAP_POS),
            "seed": args.seed,
            "objects": placements,
        }, indent=2), encoding='utf-8')
        print(f"[INFO] Wrote object positions: {positions_path}")

    sx, sy, sz = args.size
    print(f"[INFO] Generated: {args.output}")
    print(f"[INFO] Heightmap: {heightmap_path} -> {sx}x{sy}x{sz} m centered at {HEIGHTMAP_POS}")
    print(f"[INFO] Objects: {'disabled' if args.no_objects else 'enabled'}")


if __name__ == "__main__":
    main()
