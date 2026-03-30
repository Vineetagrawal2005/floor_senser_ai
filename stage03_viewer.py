import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# --- CONFIGURATION ---
WALL_HEIGHT = 45       
GLASS_COLOR = '#4ed6e8' 
GLASS_OPACITY = 0.3    
GLASS_HEADER = 40      
GLASS_SILL = 12        

# 1. LOAD DATA
try:
    with open('floorplan_3d_data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find floorplan_3d_data.json. Run Stage 02 first!")
    exit()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Stage 03: 3D Geometry with Window Cutouts", fontsize=16)

def create_poly(coords, color, alpha):
    """Helper to render a 3D polygon"""
    poly = Poly3DCollection(coords, facecolors=color, edgecolors='black', linewidths=0.3, alpha=alpha)
    ax.add_collection3d(poly)

def find_windows_on_wall(p1, p2, windows_list, margin=15):
    """Detects if a window sits on the current wall segment"""
    w_on_wall = []
    is_vert = abs(p1[1]-p2[1]) > abs(p1[0]-p2[0])
    
    for win in windows_list:
        if is_vert:
            # Check if window X overlaps wall X and window Y is within wall Y range
            if abs(win['x'] - p1[0]) < margin:
                w_on_wall.append({'start': win['y'], 'end': win['y'] + win['h']})
        else:
            # Check if window Y overlaps wall Y and window X is within wall X range
            if abs(win['y'] - p1[1]) < margin:
                w_on_wall.append({'start': win['x'], 'end': win['x'] + win['w']})
    return sorted(w_on_wall, key=lambda i: i['start'])

def make_wall_geometry(start, end, static, is_v, low, high):
    """Creates the 4 corners of a vertical rectangular plane"""
    if end <= start: return None
    if is_v:
        return [[(static, start, low), (static, end, low), (static, end, high), (static, start, high)]]
    else:
        return [[(start, static, low), (end, static, low), (end, static, high), (start, static, high)]]

def extrude_complex_wall(p1, p2, color, all_windows):
    """Draws walls with window holes and inserts cyan glass"""
    is_vertical = abs(p1[1]-p2[1]) > abs(p1[0]-p2[0])
    static_coord = p1[1-is_vertical]
    
    c_start, c_end = (p1[is_vertical], p2[is_vertical])
    if c_start > c_end: c_start, c_end = c_end, c_start
    
    cuts = find_windows_on_wall(p1, p2, all_windows)
    cursor = c_start
    
    for cut in cuts:
        # 1. Solid wall segment before window
        if cut['start'] > cursor:
            poly = make_wall_geometry(cursor, cut['start'], static_coord, is_vertical, 0, WALL_HEIGHT)
            if poly: create_poly(poly, color, 0.9)
        
        # 2. Window Area (Header, Sill, and Glass)
        w_s, w_e = max(c_start, cut['start']), min(c_end, cut['end'])
        
        # Header (Top wall)
        h_poly = make_wall_geometry(w_s, w_e, static_coord, is_vertical, GLASS_HEADER, WALL_HEIGHT)
        if h_poly: create_poly(h_poly, color, 0.9)
        
        # Sill (Bottom wall)
        s_poly = make_wall_geometry(w_s, w_e, static_coord, is_vertical, 0, GLASS_SILL)
        if s_poly: create_poly(s_poly, color, 0.9)
        
        # Glass (Transparent Cyan)
        g_poly = make_wall_geometry(w_s, w_e, static_coord, is_vertical, GLASS_SILL, GLASS_HEADER)
        if g_poly: create_poly(g_poly, GLASS_COLOR, GLASS_OPACITY)
        
        cursor = w_e

    # 3. Final solid wall segment after all windows
    if cursor < c_end:
        poly = make_wall_geometry(cursor, c_end, static_coord, is_vertical, 0, WALL_HEIGHT)
        if poly: create_poly(poly, color, 0.9)

# --- EXECUTION ---
max_x, max_y = 0, 0
windows = data.get("windows", [])

for wall in data.get("load_bearing_walls", []):
    extrude_complex_wall((wall["x1"], wall["y1"]), (wall["x2"], wall["y2"]), '#e63946', windows)
    max_x = max(max_x, wall["x1"], wall["x2"])
    max_y = max(max_y, wall["y1"], wall["y2"])

for wall in data.get("partition_walls", []):
    extrude_complex_wall((wall["x1"], wall["y1"]), (wall["x2"], wall["y2"]), '#e9c46a', windows)
    max_x = max(max_x, wall["x1"], wall["x2"])
    max_y = max(max_y, wall["y1"], wall["y2"])

# Camera/Scene setup
ax.set_xlim(0, max_x + 20)
ax.set_ylim(0, max_y + 20)
ax.set_zlim(0, WALL_HEIGHT + 10)
ax.invert_yaxis()
ax.set_box_aspect([1, (max_y/max_x), 0.3]) 
ax.axis('off') # Clean architectural look

print("3D Model Rendered. Click and drag to rotate!")
plt.show()