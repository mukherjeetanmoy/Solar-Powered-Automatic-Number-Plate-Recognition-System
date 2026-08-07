
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
from scipy.interpolate import interp1d




def interpolate_bboxes(data):
    car_ids = set()
    for frame in data:
        for car_id in data[frame]:
            car_ids.add(car_id)

    interpolated = {}
    for car_id in car_ids:
        frame_numbers = sorted([f for f in data if car_id in data[f]])
        if len(frame_numbers) < 2:
            for f in frame_numbers:
                if f not in interpolated:
                    interpolated[f] = {}
                interpolated[f][car_id] = data[f][car_id]
            continue

        # Get best plate text (highest score)
        best_text = ''
        best_score = 0
        for f in frame_numbers:
            score = data[f][car_id]['license_plate']['text_score']
            if score > best_score:
                best_score = score
                best_text = data[f][car_id]['license_plate']['text']

        car_bboxes = np.array([data[f][car_id]['car']['bbox'] for f in frame_numbers])
        plate_bboxes = np.array([data[f][car_id]['license_plate']['bbox'] for f in frame_numbers])

        full_range = range(frame_numbers[0], frame_numbers[-1] + 1)

        interp_car = interp1d(frame_numbers, car_bboxes, axis=0, kind='linear')
        interp_plate = interp1d(frame_numbers, plate_bboxes, axis=0, kind='linear')

        for f in full_range:
            if f not in interpolated:
                interpolated[f] = {}
            interpolated[f][car_id] = {
                'car': {'bbox': interp_car(f).tolist()},
                'license_plate': {
                    'bbox': interp_plate(f).tolist(),
                    'text': best_text,
                    'bbox_score': best_score,
                    'text_score': best_score
                }
            }
    return interpolated

if __name__ == '__main__':
    data = load_csv('./test.csv')
    interpolated = interpolate_bboxes(data)

    # Save interpolated data
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
              'license_plate_bbox_score', 'license_number', 'license_number_score']

    with open('./test_interpolated.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame in sorted(interpolated.keys()):
            for car_id in interpolated[frame]:
                d = interpolated[frame][car_id]
                writer.writerow([
                    frame, car_id,
                    d['car']['bbox'],
                    d['license_plate']['bbox'],
                    d['license_plate']['bbox_score'],
                    d['license_plate']['text'],
                    d['license_plate']['text_score']
                ])