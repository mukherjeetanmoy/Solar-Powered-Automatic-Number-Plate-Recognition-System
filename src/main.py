from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort import *
from util import get_car, read_license_plate, write_csv
import json
from datetime import datetime
from collections import Counter

# Map COCO class IDs to vehicle types
vehicle_names = {3: 'Car', 2: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Add a separate list for dashboard data
dashboard_data = []

detected_plates = set()

# ---- NEW: OCR stabilization buffer (for a STATIC plate) ----
# We collect the last few OCR readings and only "confirm" a plate once
# the same (or very similar) text has appeared a minimum number of times.
# This fixes noisy single-frame OCR errors without needing a moving-vehicle setup.
ocr_buffer = []          # recent raw OCR readings
BUFFER_SIZE = 8          # how many recent readings to remember (smaller = switches to a new plate faster during demo)
MIN_VOTES_TO_CONFIRM = 3 # how many times the same reading must appear to be trusted

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate.pt')
coco_model.to('cpu')
license_plate_detector.to('cpu')

# load video
cap = cv2.VideoCapture("http://10.195.6.73:5000/video_feed")
#cap = cv2.VideoCapture("sample.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
cv2.namedWindow('ANPR', cv2.WINDOW_NORMAL)

# ---- NEW: process far fewer frames since the plate/scene is static ----
PROCESS_EVERY_N_FRAMES = 15   # was 5 — static scene doesn't need frequent re-detection

while ret:
    frame_nmr += 1

    if frame_nmr % PROCESS_EVERY_N_FRAMES != 0:
        ret, frame = cap.read()
        continue

    ret, frame = cap.read()

    if ret:
        # ---- NEW: downscale frame before detection to cut CPU load ----
        frame = cv2.resize(frame, (640, 480))

        results[frame_nmr] = {}

        # detect vehicles
        detections = coco_model(frame, imgsz=320, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame, imgsz=320, verbose=False)[0]
        print("Plates detected:", len(license_plates.boxes))

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # NEW: if no full vehicle body was detected around the plate
            # (e.g. testing with just a plate image, no car body in frame),
            # fall back to using the plate's own bbox instead of skipping.
            if car_id == -1:
                xcar1, ycar1, xcar2, ycar2 = x1, y1, x2, y2

            if True:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print("OCR=", license_plate_text)

                if license_plate_text is not None:
                    # ---- NEW: add to voting buffer instead of trusting immediately ----
                    ocr_buffer.append((license_plate_text, license_plate_text_score))
                    if len(ocr_buffer) > BUFFER_SIZE:
                        ocr_buffer.pop(0)

                    texts_only = [t for t, s in ocr_buffer]
                    most_common_text, count = Counter(texts_only).most_common(1)[0]

                    print(f"Voting buffer: {texts_only}  -> leading guess: {most_common_text} ({count} votes)")

                    if count >= MIN_VOTES_TO_CONFIRM and most_common_text not in detected_plates:
                        # confirmed reading — log it once
                        avg_score = np.mean([s for t, s in ocr_buffer if t == most_common_text])
                        status = "Registered" if most_common_text.startswith("WB") else "Not Registered"

                        print("\n==========================")
                        print("CONFIRMED Plate :", most_common_text)
                        print("Status:", status)
                        print("==========================\n")

                        results[frame_nmr][0] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': most_common_text,
                                'bbox_score': score,
                                'text_score': avg_score
                            }
                        }

                        detected_plates.add(most_common_text)
                        dashboard_data.append({
                            'plate': most_common_text,
                            'status': status,
                            'confidence': round(float(avg_score), 2),
                            'vehicle_type': 'car',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'frame': frame_nmr
                        })

                        with open("detections.json", "w") as f:
                            json.dump(dashboard_data, f, indent=2)

                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    label = most_common_text if count >= MIN_VOTES_TO_CONFIRM else license_plate_text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.rectangle(frame,
                                  (int(x1), int(y1) - text_size[1] - 15),
                                  (int(x1) + text_size[0], int(y1)),
                                  (0, 255, 0), -1)
                    cv2.putText(frame, label,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        frame_resized = cv2.resize(frame, (960, 720))
        cv2.imshow('ANPR', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit early
            break

with open('detections.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

write_csv(results, './test.csv')

cap.release()
cv2.destroyAllWindows()