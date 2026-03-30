import cv2
import numpy as np
import json # <-- NEW IMPORT

# 1. LOAD AND PRE-PROCESS
img = cv2.imread('new.png', 0) 
if img is None: 
    print("Error loading image!")
    exit()

_, wall_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
_, window_thresh = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY) 

# 2. WINDOW DETECTION
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window_thresh, connectivity=8)
final_window_mask = np.zeros_like(img)
for i in range(1, num_labels):
    w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    if 2.0 < aspect_ratio < 80 and 50 < area < 3000:
        final_window_mask[labels == i] = 255

# 3. WALL CLEANING
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_thresh, connectivity=8)
clean_walls = np.zeros_like(img)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 80: 
        clean_walls[labels == i] = 255

clean_walls = cv2.subtract(clean_walls, final_window_mask)

# 4. FIND THE "OUTER SHELL"
coords = cv2.findNonZero(clean_walls)
x, y, w, h = cv2.boundingRect(coords)
global_x_min, global_x_max = x, x + w
global_y_min, global_y_max = y, y + h
margin = 25 

# 5. SKELETONIZATION
skel = np.zeros(clean_walls.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
temp_thresh = clean_walls.copy()
while True:
    eroded = cv2.erode(temp_thresh, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(temp_thresh, temp)
    skel = cv2.bitwise_or(skel, temp)
    temp_thresh = eroded.copy()
    if cv2.countNonZero(temp_thresh) == 0: break

# --- STAGE 03: PREPARE DATA DICTIONARY ---
export_data = {
    "load_bearing_walls": [],
    "partition_walls": [],
    "windows": []
}

# 6. LINE CLASSIFICATION & EXPORT
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(skel, 1, np.pi/180, 20, minLineLength=10, maxLineGap=30)

if lines is not None:
    thicknesses = []
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        if (angle < 10) or (80 < angle < 100) or (170 < angle < 190):
            mid = ((x1+x2)//2, (y1+y2)//2)
            roi = clean_walls[max(0,mid[1]-5):mid[1]+5, max(0,mid[0]-5):mid[0]+5]
            thickness = np.sum(roi == 255)
            thicknesses.append(thickness)
            valid_lines.append((x1, y1, x2, y2, thickness))
            
    dynamic_threshold = np.median(thicknesses) + 2 if thicknesses else 50
        
    for x1, y1, x2, y2, thickness in valid_lines:
        is_exterior = False
        if (abs(y1 - global_y_min) < margin and abs(y2 - global_y_min) < margin) or \
           (abs(y1 - global_y_max) < margin and abs(y2 - global_y_max) < margin) or \
           (abs(x1 - global_x_min) < margin and abs(x2 - global_x_min) < margin) or \
           (abs(x1 - global_x_max) < margin and abs(x2 - global_x_max) < margin):
               is_exterior = True

        # Note: Casting to int() is required because JSON cannot read numpy data types
        line_data = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

        if is_exterior or thickness >= dynamic_threshold:
            cv2.line(output, (x1,y1), (x2,y2), (0,0,255), 3) 
            export_data["load_bearing_walls"].append(line_data)
        else:
            cv2.line(output, (x1,y1), (x2,y2), (0,255,255), 2) 
            export_data["partition_walls"].append(line_data)

# 7. DRAW WINDOWS & EXPORT
cnts, _ = cv2.findContours(final_window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output, (x,y), (x+w, y+h), (255, 255, 0), 2)
    export_data["windows"].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

# --- STAGE 03: SAVE TO JSON ---
with open('floorplan_3d_data.json', 'w') as json_file:
    json.dump(export_data, json_file, indent=4)
print("SUCCESS: floorplan_3d_data.json has been generated!")

cv2.imshow("Topology Validated Graph", output)
cv2.waitKey(0)
cv2.destroyAllWindows()