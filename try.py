import json
import base64
import os

def base64_to_png(base64_string, output_path):
    """
    Decodes a base64 string and saves it as a PNG file.

    :param base64_string: The base64 encoded string of the image.
    :param output_path: The file path where the PNG will be saved.
    """
    if not output_path.lower().endswith('.png'):
        raise ValueError("Output file must have a .png extension.")
    
    # Remove the data URL scheme if present
    if base64_string.startswith("data:image/png;base64,"):
        base64_string = base64_string.replace("data:image/png;base64,", "")
    
    try:
        image_data = base64.b64decode(base64_string)
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 string: {e}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, "wb") as output_file:
            output_file.write(image_data)
        print(f"Saved PNG: {output_path}")
    except IOError as e:
        raise IOError(f"Failed to write image to {output_path}: {e}")

def convert_json_to_pngs(json_file_path, output_directory):
    """
    Reads a JSON file containing a list of objects with base64-encoded PNG images
    and saves each image as a separate PNG file.

    :param json_file_path: Path to the JSON file.
    :param output_directory: Directory where PNG files will be saved.
    """
    # Read the JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Ensure data is a list
    if not isinstance(data, list):
        print("JSON data is not a list of objects.")
        return

    # Iterate through each object and save the PNG
    counter = 1
    for item in data:
        counter += 1
        try:
            question_number = item.get("question_number")
            base64_string = item.get("question_selector")

            if question_number is None or base64_string is None:
                print("Missing 'question_number' or 'question_selector' in item:", item)
                continue

            # Define the output file name, e.g., question_1.png
            output_filename = f"{counter}.png"
            output_path = os.path.join(output_directory, output_filename)

            # Convert and save the PNG
            base64_to_png(base64_string, output_path)

        except Exception as e:
            print(f"Error processing item {item}: {e}")

if __name__ == "__main__":
    # # Path to your JSON file
    # json_file = "train_section_data2.json"

    # # Directory where you want to save the PNG files
    # output_dir = "help"

    # # Convert the JSON data to PNG files
    # convert_json_to_pngs(json_file, output_dir)
    with open("b64.txt", 'r') as f:
        b = f.read()
    base64_to_png(b, "testing/hoi.png")
