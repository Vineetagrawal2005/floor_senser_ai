import cv2
import numpy as np


img = cv2.imread('FLOOR.png', 0) 
if img is None: exit()

_, wall_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
_, window_thresh = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY) 


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window_thresh, connectivity=8)
final_window_mask = np.zeros_like(img)
for i in range(1, num_labels):
    w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    
    
    if 2.0 < aspect_ratio < 80 and 50 < area < 3000:
        final_window_mask[labels == i] = 255


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_thresh, connectivity=8)
clean_walls = np.zeros_like(img)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 80: 
        clean_walls[labels == i] = 255

clean_walls = cv2.subtract(clean_walls, final_window_mask)


coords = cv2.findNonZero(clean_walls)
x, y, w, h = cv2.boundingRect(coords)
global_x_min, global_x_max = x, x + w
global_y_min, global_y_max = y, y + h
margin = 25 # Pixel tolerance for defining "the edge"

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

# 6. DYNAMIC LINE CLASSIFICATION + EXTERIOR OVERRIDE
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
        # EXTERIOR OVERRIDE LOGIC: If a line is near the absolute edge of the house, force it to RED
        is_exterior = False
        if (abs(y1 - global_y_min) < margin and abs(y2 - global_y_min) < margin) or \
           (abs(y1 - global_y_max) < margin and abs(y2 - global_y_max) < margin) or \
           (abs(x1 - global_x_min) < margin and abs(x2 - global_x_min) < margin) or \
           (abs(x1 - global_x_max) < margin and abs(x2 - global_x_max) < margin):
               is_exterior = True

        if is_exterior or thickness >= dynamic_threshold:
            cv2.line(output, (x1,y1), (x2,y2), (0,0,255), 3) # RED (Load-bearing)
        else:
            cv2.line(output, (x1,y1), (x2,y2), (0,255,255), 2) # YELLOW (Partition)

# 7. DRAW WINDOWS (Cyan)
cnts, _ = cv2.findContours(final_window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output, (x,y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow("Topology Validated Graph", output)
cv2.waitKey(0)
cv2.destroyAllWindows()