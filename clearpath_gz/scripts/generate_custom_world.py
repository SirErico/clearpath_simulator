"""Generate a Gazebo SDF world from a greyscale heightmap.

The heightmap is used both as the Gazebo terrain (SDF <heightmap> geometry)
and as the elevation source for tree placement. World coordinates:
  - heightmap is centered at HEIGHTMAP_POS
  - X spans [pos_x - size_x/2, pos_x + size_x/2]
  - Y spans [pos_y - size_y/2, pos_y + size_y/2]
  - Z = (pixel/255) * size_z + pos_z
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

DEFAULT_HEIGHTMAP = PACKAGE_DIR / 'heightmaps' / 'hills1_513.png'
DEFAULT_OUTPUT = PACKAGE_DIR / 'worlds' / 'forest_world.sdf'
# Solid-color texture used as the heightmap diffuse so per-patch tiling is invisible.
SOLID_TEXTURE = PACKAGE_DIR / 'heightmaps' / 'forest_texture.png'

TREE_MODEL_URI = "model://Pine Tree"

# Heightmap world placement (meters). Centered at origin by default.
HEIGHTMAP_POS = (0.0, 0.0, 0.0)

# Forest layout
TREE_COUNT = 16
TREE_EDGE_MARGIN = 0.5
TREE_MAX_ATTEMPTS_FACTOR = 20
TREE_GROUND_PENETRATION = 0.08
# Skip trees where local slope exceeds this (rise/run, dimensionless).
MAX_TREE_SLOPE = 0.5

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


def generate_tree_block(heightmap: np.ndarray, size, pos, tree_count: int, seed=None):
    """Place trees at random valid points sampled from the heightmap footprint."""
    rng = np.random.default_rng(seed)
    trees = []
    skipped_spawn = 0
    skipped_slope = 0
    attempts = 0
    max_attempts = max(1, tree_count * TREE_MAX_ATTEMPTS_FACTOR)
    clearance_sq = ROBOT_CLEARANCE_RADIUS ** 2
    cx, cy = pos[0], pos[1]
    half_x = size[0] / 2
    half_y = size[1] / 2

    min_x = cx - half_x + TREE_EDGE_MARGIN
    max_x = cx + half_x - TREE_EDGE_MARGIN
    min_y = cy - half_y + TREE_EDGE_MARGIN
    max_y = cy + half_y - TREE_EDGE_MARGIN

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Heightmap area too small after applying TREE_EDGE_MARGIN.")

    while len(trees) < tree_count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)

        if (x - ROBOT_SPAWN_X) ** 2 + (y - ROBOT_SPAWN_Y) ** 2 < clearance_sq:
            skipped_spawn += 1
            continue

        if local_slope(heightmap, x, y, size, pos) > MAX_TREE_SLOPE:
            skipped_slope += 1
            continue

        z = sample_elevation(heightmap, x, y, size, pos) - TREE_GROUND_PENETRATION
        yaw = rng.uniform(-np.pi, np.pi)
        tree_id = len(trees)
        trees.append(f"""
    <include>
      <uri>{TREE_MODEL_URI}</uri>
      <name>tree_{tree_id}</name>
      <pose>{x:.2f} {y:.2f} {z:.3f} 0 0 {yaw:.3f}</pose>
    </include>""")

    print(f"[INFO] Trees placed: {len(trees)}/{tree_count} | attempts: {attempts}/{max_attempts}")
    print(f"[INFO] Trees skipped | spawn: {skipped_spawn} | slope: {skipped_slope}")
    return "\n".join(trees)


def generate_terrain(heightmap_path: Path, size, pos):
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
                <size>1</size>
                <diffuse>file://{SOLID_TEXTURE}</diffuse>
                <normal>file://{SOLID_TEXTURE}</normal>
              </texture>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>"""


def generate_world(heightmap_path: Path, size, pos, with_trees: bool, tree_count: int, seed=None):
    heightmap = load_heightmap(heightmap_path)
    terrain_block = generate_terrain(heightmap_path, size, pos)
    tree_block = (
        generate_tree_block(heightmap, size, pos, tree_count, seed=seed)
        if with_trees
        else "    <!-- trees disabled -->"
    )

    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="forest_world">

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <physics name="default_physics" type="ode">
      <max_step_size>0.003</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>39.512152</latitude_deg>
      <longitude_deg>22.426669</longitude_deg>
      <elevation>344</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1.0 1.0 1.0 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.5 -1.0</direction>
    </light>

    <scene>
      <ambient>1 1 1 1</ambient>
      <background>0.3 0.7 0.9 1</background>
      <shadows>1</shadows>
    </scene>

    {terrain_block}

    <!-- Pine Trees -->
    {tree_block}

  </world>
</sdf>
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heightmap', type=Path, default=DEFAULT_HEIGHTMAP,
                   help='Path to greyscale heightmap PNG (square, side = 2^n + 1).')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                   help='Output SDF path.')
    p.add_argument('--size', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   default=[20.0, 20.0, 1.0],
                   help='World extents of the heightmap in meters.')
    p.add_argument('--tree-count', type=int, default=TREE_COUNT,
                   help='Number of trees to attempt to place randomly.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducible tree placement.')
    p.add_argument('--no-trees', action='store_true',
                   help='Skip tree placement (terrain-only world for testing).')
    return p.parse_args()


def main():
    args = parse_args()
    heightmap_path = args.heightmap.resolve()
    if not heightmap_path.exists():
        raise SystemExit(f"Heightmap not found: {heightmap_path}")
    ensure_solid_texture(SOLID_TEXTURE)

    if args.tree_count < 0:
        raise SystemExit("--tree-count must be >= 0")

    sdf = generate_world(heightmap_path, tuple(args.size), HEIGHTMAP_POS,
                         with_trees=not args.no_trees,
                         tree_count=args.tree_count,
                         seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sdf, encoding='utf-8')

    sx, sy, sz = args.size
    print(f"[INFO] Generated: {args.output}")
    print(f"[INFO] Heightmap: {heightmap_path} -> {sx}x{sy}x{sz} m centered at {HEIGHTMAP_POS}")
    print(f"[INFO] Trees: {'disabled' if args.no_trees else 'enabled'}")


if __name__ == "__main__":
    main()
