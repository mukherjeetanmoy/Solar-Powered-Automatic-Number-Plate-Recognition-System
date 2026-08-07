import cv2
import csv
import numpy as np
import ast

def load_csv(filepath):
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame_nmr'])
            car_id = int(float(row['car_id']))
            if frame not in results:
                results[frame] = {}

            # Parse bbox strings like "[x1 y1 x2 y2]" or "[x1, y1, x2, y2]"
            car_bbox = row['car_bbox'].replace(' ', ', ') if ', ' not in row['car_bbox'] else row['car_bbox']
            plate_bbox = row['license_plate_bbox'].replace(' ', ', ') if ', ' not in row['license_plate_bbox'] else row['license_plate_bbox']

            car_bbox = [float(x) for x in ast.literal_eval(car_bbox)]
            plate_bbox = [float(x) for x in ast.literal_eval(plate_bbox)]

            results[frame][car_id] = {
                'car': {'bbox': car_bbox},
                'license_plate': {
                    'bbox': plate_bbox,
                    'text': row['license_number'],
                    'bbox_score': float(row['license_plate_bbox_score']),
                    'text_score': float(row['license_number_score'])
                }
            }
    return results



# Load interpolated results
results = load_csv('./test_interpolated.csv')

# Load video
cap = cv2.VideoCapture('./sample.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

cv2.namedWindow('ANPR Output', cv2.WINDOW_NORMAL)

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr in results:
        for car_id in results[frame_nmr]:
            d = results[frame_nmr][car_id]

            # Car bbox coordinates
            cx1, cy1, cx2, cy2 = [int(v) for v in d['car']['bbox']]
            # Plate bbox coordinates
            px1, py1, px2, py2 = [int(v) for v in d['license_plate']['bbox']]
            plate_text = d['license_plate']['text']

            # --- Draw GREEN box around car ---
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)

            # --- Draw RED box around license plate ---
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

            # --- Crop the license plate from the frame ---
            plate_crop = frame[py1:py2, px1:px2]

            # --- White banner above the car ---
            banner_h = 120
            banner_w = cx2 - cx1
            if banner_w < 200:
                banner_w = 200

            # Position the banner above the car box
            bx1 = cx1
            by1 = cy1 - banner_h - 10
            if by1 < 0:
                by1 = 0

            # Draw white background
            cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h),
                          (255, 255, 255), -1)
            # Black border
            cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h),
                          (0, 0, 0), 2)

            # Put the cropped plate image on the top half
            if plate_crop.size > 0:
                crop_h = banner_h // 2
                crop_w = banner_w - 20
                if crop_w > 0 and crop_h > 0:
                    plate_resized = cv2.resize(plate_crop, (crop_w, crop_h))
                    frame[by1 + 5:by1 + 5 + crop_h, bx1 + 10:bx1 + 10 + crop_w] = plate_resized

            # Put plate text on the bottom half
            text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = bx1 + (banner_w - text_size[0]) // 2
            text_y = by1 + banner_h - 15
            cv2.putText(frame, plate_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Write and display
    out.write(frame)
    cv2.imshow('ANPR Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('Output saved to out.mp4')