
# check_csv.py
import csv

with open('test.csv', 'r') as f:
    reader = list(csv.DictReader(f))
    print(f"Total detections in test.csv: {len(reader)}")
    if reader:
        frames = [int(r['frame_nmr']) for r in reader]
        print(f"Frame range: {min(frames)} to {max(frames)}")
        print(f"Unique cars: {len(set(r['car_id'] for r in reader))}")

with open('test_interpolated.csv', 'r') as f:
    reader = list(csv.DictReader(f))
    print(f"\nTotal detections in test_interpolated.csv: {len(reader)}")
    if reader:
        frames = [int(r['frame_nmr']) for r in reader]
        print(f"Frame range: {min(frames)} to {max(frames)}")

