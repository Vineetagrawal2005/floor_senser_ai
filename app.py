"""
FloorSense AI — Backend (app.py)
Flask server that accepts a floor plan image, runs OpenCV analysis,
and returns the processed image URL + structured wall/window JSON.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_file('templates/index.html')


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve processed images from the outputs directory."""
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/upload', methods=['POST'])
def upload():
    """
    POST /upload
    Accepts a multipart form with a 'file' field (PNG/JPG floor plan).
    Returns JSON:
      {
        "success": true,
        "job_id": "...",
        "processed_image_url": "/outputs/xxx_analyzed.png",
        "wall_data": { load_bearing_walls, partition_walls, windows },
        "stats": { load_bearing_count, partition_count, window_count }
      }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Save the raw upload
    job_id = uuid.uuid4().hex[:10]
    upload_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_input.png')
    file.save(upload_path)

    try:
        result = process_floorplan(upload_path, job_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(result)


# ─── Core OpenCV Pipeline ───────────────────────────────────────────────────────

def process_floorplan(image_path: str, job_id: str) -> dict:
    """
    Full pipeline (ported directly from stage02_extractor.py):
      1. Threshold → separate walls and windows
      2. Clean wall blobs via connected components
      3. Skeletonize walls
      4. HoughLinesP → classify lines as load-bearing vs. partition
      5. Detect windows
      6. Save annotated image
    Returns a dict ready to be serialised as JSON.
    """

    # 1. LOAD AND PRE-PROCESS ─────────────────────────────────────────────────
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Could not read the image. Please upload a valid PNG or JPG file.")

    _, wall_thresh   = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    _, window_thresh = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)

    # 2. WINDOW DETECTION ─────────────────────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        window_thresh, connectivity=8
    )
    final_window_mask = np.zeros_like(img)
    for i in range(1, num_labels):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if 2.0 < aspect_ratio < 80 and 50 < area < 3000:
            final_window_mask[labels == i] = 255

    # 3. WALL CLEANING ────────────────────────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wall_thresh, connectivity=8
    )
    clean_walls = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 80:
            clean_walls[labels == i] = 255

    clean_walls = cv2.subtract(clean_walls, final_window_mask)

    # 4. FIND THE "OUTER SHELL" ───────────────────────────────────────────────
    coords = cv2.findNonZero(clean_walls)
    if coords is None:
        raise ValueError(
            "No wall structures detected. "
            "Try a higher-contrast floor plan with clear dark walls on a white background."
        )
    x, y, w, h = cv2.boundingRect(coords)
    global_x_min, global_x_max = x, x + w
    global_y_min, global_y_max = y, y + h
    margin = 25

    # 5. SKELETONIZATION ──────────────────────────────────────────────────────
    skel    = np.zeros(clean_walls.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp    = clean_walls.copy()
    while True:
        eroded   = cv2.erode(temp, element)
        dilated  = cv2.dilate(eroded, element)
        diff     = cv2.subtract(temp, dilated)
        skel     = cv2.bitwise_or(skel, diff)
        temp     = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # 6. LINE DETECTION & CLASSIFICATION ─────────────────────────────────────
    export_data = {
        "load_bearing_walls": [],
        "partition_walls":    [],
        "windows":            []
    }
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=30)

    if lines is not None:
        thicknesses  = []
        valid_lines  = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Keep only axis-aligned lines (H or V within ±10°)
            if (angle < 10) or (80 < angle < 100) or (170 < angle < 190):
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                roi = clean_walls[
                    max(0, mid[1] - 5): mid[1] + 5,
                    max(0, mid[0] - 5): mid[0] + 5
                ]
                thickness = int(np.sum(roi == 255))
                thicknesses.append(thickness)
                valid_lines.append((x1, y1, x2, y2, thickness))

        dynamic_threshold = (np.median(thicknesses) + 2) if thicknesses else 50

        for x1, y1, x2, y2, thickness in valid_lines:
            is_exterior = (
                (abs(y1 - global_y_min) < margin and abs(y2 - global_y_min) < margin) or
                (abs(y1 - global_y_max) < margin and abs(y2 - global_y_max) < margin) or
                (abs(x1 - global_x_min) < margin and abs(x2 - global_x_min) < margin) or
                (abs(x1 - global_x_max) < margin and abs(x2 - global_x_max) < margin)
            )
            line_data = {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2)
            }
            if is_exterior or thickness >= dynamic_threshold:
                cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)   # Red
                export_data["load_bearing_walls"].append(line_data)
            else:
                cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Cyan/Yellow
                export_data["partition_walls"].append(line_data)

    # 7. DRAW WINDOWS ─────────────────────────────────────────────────────────
    cnts, _ = cv2.findContours(
        final_window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(output_img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
        export_data["windows"].append({
            "x": int(bx), "y": int(by),
            "w": int(bw), "h": int(bh)
        })

    # 8. SAVE ANNOTATED IMAGE ─────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_analyzed.png')
    cv2.imwrite(out_path, output_img)

    return {
        'success':              True,
        'job_id':               job_id,
        'processed_image_url':  f'/outputs/{job_id}_analyzed.png',
        'wall_data':            export_data,
        'stats': {
            'load_bearing_count': len(export_data['load_bearing_walls']),
            'partition_count':    len(export_data['partition_walls']),
            'window_count':       len(export_data['windows']),
        }
    }


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  FloorSense AI — Backend")
    print("  Running at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
