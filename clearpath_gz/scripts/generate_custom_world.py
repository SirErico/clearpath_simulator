import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
OUTPUT_FILE = PACKAGE_DIR / "worlds" / "forest_world.sdf"

TREE_MODEL_URI = "model://Pine Tree"
HEIGHTMAP_FILE = PACKAGE_DIR / 'heightmaps' / 'hills1_513.png'
HEIGHTMAP_URI = f"file://{HEIGHTMAP_FILE.as_posix()}"
TEXTURE_URI = f"file://{(PACKAGE_DIR / 'meshes' / 'forest' / 'Ground.jpg').as_posix()}"
NORMAL_URI = f"file://{(PACKAGE_DIR / 'meshes' / 'forest' / 'flat_normal.jpg').as_posix()}"

# Terrain parameters
TERRAIN_SIZE_X = 150.0
TERRAIN_SIZE_Y = 150.0
HEIGHT_SCALE_Z = 2.0  # Kept small for gentle hills

# Forest parameters
ROWS = 10
COLS = 5
SPACING_X = 3.0
SPACING_Y = 3.5
JITTER = 0.8
TREE_GROUND_PENETRATION = 0.08  # Sink trunks slightly into ground to prevent hover gaps


def get_heightmap_data(heightmap_path: Path):
    """Load the heightmap image into a numpy array."""
    image = Image.open(heightmap_path).convert('L')
    return np.array(image, dtype=np.float32)


def compute_terrain_z_offset(heights: np.ndarray, size_z: float) -> float:
    """
    Shift terrain so world origin (0,0,0) sits exactly on the terrain surface 
    at the center of the map.
    """
    center_height = heights[heights.shape[0] // 2, heights.shape[1] // 2]
    # In Gazebo, heightmap origin is at the bottom of the bounding box.
    return -(center_height / 255.0) * size_z


def get_tree_z(heights: np.ndarray, x: float, y: float, size_x: float, size_y: float, size_z: float, z_offset: float) -> float:
    """Calculate the Z position for a tree on the orchard surface."""
    # Orchard mesh and tree origins are close to z=0, but a tiny negative offset
    # helps avoid visible floating and lidar rays passing below trunks.
    return -TREE_GROUND_PENETRATION


def generate_tree_block(heights: np.ndarray, terrain_z_offset: float):
    trees = []

    for i in range(1, ROWS):
        for j in range(1, COLS):
            x = i * SPACING_X
            y = j * SPACING_Y + np.random.uniform(-JITTER, JITTER)
            yaw = np.random.uniform(-np.pi, np.pi)
            
            # Get the exact Z coordinate for this specific X, Y location
            z = get_tree_z(heights, x, y, TERRAIN_SIZE_X, TERRAIN_SIZE_Y, HEIGHT_SCALE_Z, terrain_z_offset)

            tree = f"""
    <include>
      <uri>{TREE_MODEL_URI}</uri>
      <name>tree_{i}_{j}</name>
      <pose>{x:.2f} {y:.2f} {z:.3f} 0 0 {yaw:.3f}</pose>
    </include>
"""
            trees.append(tree)

    return "\n".join(trees)


def generate_world():
    # Load heightmap for tree positioning
    heights = get_heightmap_data(HEIGHTMAP_FILE)
    terrain_z_offset = compute_terrain_z_offset(heights, HEIGHT_SCALE_Z)
    
    # Generate trees using the heightmap data
    tree_block = generate_tree_block(heights, terrain_z_offset)

    sdf = f"""<?xml version="1.0" ?>
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

    <!-- Origin placed on a random olive orchard somewhere in Greece -->
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

    <!-- Orchard surface model -->
    <model name='orchard'>
      <pose>-10 -10 0 0 0 0</pose>
      <static>true</static>
      <link name='orchard'>
        <collision name='orchard_collision'>
          <geometry>
            <mesh>
              <uri>
                model://orchard/orchard_world.dae
              </uri>
            </mesh>
          </geometry>
        </collision>
        <visual name='orchard_visual'>
          <geometry>
            <mesh>
              <uri>
                model://orchard/orchard_world.dae
              </uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Pine Trees -->
    {tree_block}

  </world>
</sdf>
"""

    return sdf


def main():
    sdf_content = generate_world()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(sdf_content)

    print(f"[INFO] Generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()