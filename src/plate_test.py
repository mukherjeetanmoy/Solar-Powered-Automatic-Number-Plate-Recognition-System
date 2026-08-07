from ultralytics import YOLO
import util
import cv2
from util import read_license_plate
model = YOLO("license_plate.pt")

cap = cv2.VideoCapture("http://10.36.139.73:5000/video_feed")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.15, verbose=False)[0]

    for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            cv2.imshow("Crop",crop)
            cv2.waitKey(0)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            text, text_score = util.read_license_plate(gray)
            print("------------")
            print("OCR:",text)
            print("Score:",score)
            print("Detected:", text, text_score)

            cv2.rectangle(frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0), 2)
            #cv2.rectangle(frame,
            #      (int(x1), int(y1)),
             #         (int(x2), int(y2)),
             #         (0, 255, 0), 2)

    cv2.imshow("Plate Test", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()