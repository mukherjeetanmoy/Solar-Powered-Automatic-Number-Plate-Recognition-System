import string
import easyocr
import re
# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Accept plates between 5-8 characters that have a mix of letters and digits.
    Much more lenient than strict UK format.
    """
    if len(text) < 5 or len(text) > 8:
        return False

    has_letter = any(c in string.ascii_uppercase for c in text)
    has_digit = any(c.isdigit() for c in text)

    return has_letter and has_digit


def format_license(text):
    """
    Clean up the plate text using common OCR corrections.
    """
    mapping = {
        'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6',
        'S': '5', 'B': '8', 'Z': '2', 'T': '7'
    }

    # Try to apply UK format AA00AAA if it fits
    if len(text) == 7:
        result = ''
        for i, char in enumerate(text):
            if i < 2 or i > 3:  # letter positions
                if char in dict_int_to_char:
                    result += dict_int_to_char[char]
                else:
                    result += char
            else:  # digit positions
                if char in dict_char_to_int:
                    result += dict_char_to_int[char]
                else:
                    result += char
        return result

    # Otherwise return cleaned text as-is
    return text


# def read_license_plate(license_plate_crop):

#     detections = reader.readtext(license_plate_crop)

#     best_text = None
#     best_score = 0

#     # for detection in detections:
#     #     bbox, text, score = detection
#     #     text = text.upper().replace(' ', '')

#     #     pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'

#     #     if not re.match(pattern, text):
#     #         continue
#     #     # Accept any text 4+ characters with confidence > 0.2
#     #     if len(text) >= 4 and score > 0.2 and score > best_score:
#     #         best_text = text
#     #         best_score = score

#     # if best_text is not None:
#     #     return best_text, best_score

#     # return None, None
#     for detection in detections:
#     bbox, text, score = detection
#     text = text.upper().replace(' ', '')

#     pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'

#     if re.match(pattern, text) and score > best_score:
#         best_text = text
#         best_score = score

#     if best_text is not None:
#     return best_text, best_score

#     return None, None
def read_license_plate(license_plate_crop):

    detections = reader.readtext(license_plate_crop)

    best_text = None
    best_score = 0

    pattern = r'^[A-Z0-9]{8,12}$'

    bad_words = [
        "HONDA",
        "HYUNDAI",
        "SUZUKI",
        "TOYOTA",
        "KIA",
        "CITY",
        "CRETA",
        "INDIAMART",
        "NUMBER",
        "PLATE"
    ]

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(" ", "")

        print("OCR RAW:", text, "Score:", score)

        if any(word in text for word in bad_words):
            continue

        if re.match(pattern, text) and score > best_score:
            best_text = text
            best_score = score

    if best_text is not None:
        return best_text, best_score

    return None, None
def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
