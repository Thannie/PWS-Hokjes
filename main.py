import torch
import cv2
import base64
import json
import numpy as np
import time
import warnings

# Suppress FutureWarnings temporarily
warnings.filterwarnings("ignore", category=FutureWarning)

EXPECTED_NUMBER_OF_CHECKBOXES = 7

# Load the model globally
MODEL_WEIGHTS = 'yolov5/runs/train/checkbox_yolov5_transfer8/weights/best.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS, force_reload=False)
model.to(DEVICE)
model.iou = 0.2

class Checkbox:
    """
    Represents a checkbox with its class ID and fill percentage.
    """
    def __init__(self, class_id, fill_percentage):
        self.class_id = class_id
        self.fill_percentage = fill_percentage

    def is_greater_than(self, percentage=4):
        """Check if the fill percentage is greater than a given value."""
        return self.fill_percentage > percentage

    def percentage_difference(self, other):
        """Calculate the percentage difference between two checkboxes."""
        if not isinstance(other, Checkbox):
            raise ValueError("Can only compare with another Checkbox object.")
        if self.fill_percentage != 0:
            return abs((other.fill_percentage - self.fill_percentage) / self.fill_percentage * 100)
        return 0

    def __str__(self):
        return f"Number: {self.class_id}, fill percentage: {self.fill_percentage}"

    # Comparison based on fill percentage
    def __lt__(self, other):
        if not isinstance(other, Checkbox):
            return NotImplemented
        return self.fill_percentage < other.fill_percentage

    def __le__(self, other):
        if not isinstance(other, Checkbox):
            return NotImplemented
        return self.fill_percentage <= other.fill_percentage

    def __eq__(self, other):
        if not isinstance(other, Checkbox):
            return NotImplemented
        return self.fill_percentage == other.fill_percentage

    def __gt__(self, other):
        if not isinstance(other, Checkbox):
            return NotImplemented
        return self.fill_percentage > other.fill_percentage

    def __ge__(self, other):
        if not isinstance(other, Checkbox):
            return NotImplemented
        return self.fill_percentage >= other.fill_percentage

def decode_base64_image(b64_string):
    """
    Decode a Base64-encoded image into an OpenCV image.

    Args:
        b64_string (str): Base64-encoded image string.

    Returns:
        numpy.ndarray: Decoded image in BGR format.
    """
    # Remove header if present (e.g., "data:image/png;base64,")
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    # Decode the Base64 string
    try:
        image_data = base64.b64decode(b64_string)
    except base64.binascii.Error as e:
        raise ValueError(f"Base64 decoding failed: {e}")

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode image with OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the image. Please check the Base64 string.")

    return image

def get_detections(model, image, img_size=512):
    """
    Perform inference on an image and retrieve detections.

    Args:
        model (torch.nn.Module): Loaded YOLOv5 model.
        image (numpy.ndarray): Input image in BGR format.
        img_size (int): Image size for inference.

    Returns:
        list: List of detections with bounding box coordinates, confidence, and class ID.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image_rgb, size=img_size)

    detections = results.xyxy[0]  # Detections for the first image
    return detections

def rank_and_filter_checkboxes(checkbox_data, expected_count, spacing_tolerance=10, size_tolerance=0.2, horizontal_tolerance=10):
    """
    Ranks and filters checkboxes based on alignment, spacing, size, and horizontal alignment.
    Also removes checkboxes that are too close vertically.

    Args:
        checkbox_data (list of tuple): List of tuples containing Checkbox objects and their (y1, x1, x2, y2) coordinates.
        expected_count (int): The target number of checkboxes to select.
        spacing_tolerance (int): Minimum vertical spacing in pixels for two checkboxes to be considered separate.
        size_tolerance (float): Tolerance for size variation as a fraction of average size.
        horizontal_tolerance (int): Tolerance for horizontal deviation in pixels.

    Returns:
        list of Checkbox: Filtered and ranked list of Checkbox objects.
    """
    if not checkbox_data:
        raise ValueError("No checkbox data provided.")

    # Extract properties for calculations
    centers_x = [(x1 + x2) / 2 for _, y1, x1, x2, y2 in checkbox_data]
    heights = [y2 - y1 for _, y1, x1, x2, y2 in checkbox_data]
    widths = [x2 - x1 for _, y1, x1, x2, y2 in checkbox_data]

    # Calculate averages
    mean_x_center = np.mean(centers_x)
    mean_height = np.mean(heights)
    mean_width = np.mean(widths)

    # Sort by y1 (top-to-bottom order)
    sorted_checkboxes = sorted(checkbox_data, key=lambda x: x[1])

    # Filter out checkboxes that are too close vertically
    filtered_checkboxes = []
    last_y = None

    for checkbox, y1, x1, x2, y2 in sorted_checkboxes:
        if last_y is None or abs(y1 - last_y) > spacing_tolerance:
            filtered_checkboxes.append((checkbox, y1, x1, x2, y2))
            last_y = y1  # Update the last y value

    # Rank remaining checkboxes by likelihood score
    ranked_checkboxes = []
    for checkbox, y1, x1, x2, y2 in filtered_checkboxes:
        # Calculate individual scores
        center_x = (x1 + x2) / 2
        width = x2 - x1
        height = y2 - y1

        # Horizontal alignment score
        horizontal_score = 1 if abs(center_x - mean_x_center) <= horizontal_tolerance else 0

        # Size consistency score
        size_score = (
            1 if abs(width - mean_width) / mean_width <= size_tolerance and
                 abs(height - mean_height) / mean_height <= size_tolerance
            else 0
        )

        # Total likelihood score
        likelihood_score = horizontal_score + size_score
        ranked_checkboxes.append((checkbox, likelihood_score))

    # Sort by likelihood score in descending order
    ranked_checkboxes = sorted(ranked_checkboxes, key=lambda x: x[1], reverse=True)

    # Select the top-ranked checkboxes based on the expected count
    selected_checkboxes = [cb for cb, _ in ranked_checkboxes[:expected_count]]

    return selected_checkboxes


def analyze_fill_level(checkbox_img, threshold=127, fixed_margin=5):
    """
    Analyze the fill level of a checkbox by excluding a fixed border margin.

    Args:
        checkbox_img (numpy.ndarray): Cropped image of the checkbox (BGR format).
        threshold (int): Threshold for binary conversion (0-255).
        fixed_margin (int): Fixed number of pixels to exclude as margin.

    Returns:
        float: Fill level as a percentage (0 to 100).
    """
    # Validate image
    if checkbox_img.size == 0:
        raise ValueError("Cropped checkbox image is empty.")

    height, width = checkbox_img.shape[:2]

    # Ensure margin does not exceed image dimensions
    if height <= 2 * fixed_margin or width <= 2 * fixed_margin:
        raise ValueError("Fixed margin is too large, resulting in empty cropped image.")

    # Exclude the fixed margin from all sides
    cropped_img = checkbox_img[fixed_margin:height - fixed_margin, fixed_margin:width - fixed_margin]

    # Convert to grayscale
    try:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        raise ValueError(f"cv2.cvtColor failed: {e}")

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Calculate the percentage of non-zero pixels (filled area)
    non_zero_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    fill_percentage = (non_zero_pixels / total_pixels) * 100

    return fill_percentage

def process_image_data(image, img_size=512, threshold=127, margin_ratio=0.05):
    """
    Processes an image to detect and filter checkboxes.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        img_size (int, optional): Image size for inference. Defaults to 512.
        threshold (int, optional): Threshold for binary conversion. Defaults to 127.
        margin_ratio (float, optional): Ratio of image dimensions to use as margin. Defaults to 0.05.

    Returns:
        list: List of Checkbox objects.
    """
    # Perform inference
    detections = get_detections(model, image, img_size)

    # List to store tuples of (Checkbox, y1, x1, x2, y2)
    checkbox_data = []

    # Iterate over detections
    for *xyxy, conf, cls in detections:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, xyxy)

        # Validate bounding box dimensions
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop the checkbox region from the image
        checkbox_img = image[y1:y2, x1:x2]

        # Analyze fill level with margin exclusion
        try:
            fill_percentage = analyze_fill_level(checkbox_img, threshold)
        except ValueError:
            continue

        # Create a Checkbox object
        checkbox = Checkbox(class_id=None, fill_percentage=fill_percentage)

        # Append the checkbox and its coordinates for filtering
        checkbox_data.append((checkbox, y1, x1, x2, y2))

    if not checkbox_data:
        raise ValueError("No valid checkboxes detected in the image.")

    # Check for excess checkboxes
    if len(checkbox_data) > EXPECTED_NUMBER_OF_CHECKBOXES * 1.3:
        raise ValueError(f"Too many checkboxes detected: {len(checkbox_data)}. Expected around {EXPECTED_NUMBER_OF_CHECKBOXES}.")

    # Rank and filter checkboxes
    filtered_checkboxes = rank_and_filter_checkboxes(
        checkbox_data,
        expected_count=EXPECTED_NUMBER_OF_CHECKBOXES,
        spacing_tolerance=10,
        size_tolerance=0.2,
        horizontal_tolerance=10
    )

    # Assign class IDs based on sorted order
    for idx, checkbox in enumerate(filtered_checkboxes, start=1):
        checkbox.class_id = idx

    return filtered_checkboxes

def analyze_top_checkboxes(checkboxes, percentage_threshold=5, individual_threshold=4):
    """
    Analyze the two highest-filled checkboxes for specific conditions.

    Args:
        checkboxes (list of Checkbox): List of Checkbox objects.
        percentage_threshold (float): Minimum percentage difference to check between top two.
        individual_threshold (float): Minimum individual fill percentage.

    Returns:
        tuple: (analysis_results_dict, sorted_checkboxes)
    """
    # Sort checkboxes by fill percentage in descending order
    sorted_checkboxes = sorted(checkboxes, reverse=True)

    # Get the top two checkboxes
    if len(sorted_checkboxes) < 2:
        raise ValueError("At least two checkboxes are required for this analysis.")

    top1 = sorted_checkboxes[0]
    top2 = sorted_checkboxes[1]

    # Calculate the percentage difference
    difference = top1.percentage_difference(top2)

    # Check the conditions
    results = {
        "difference_greater_than_threshold": difference > percentage_threshold,
        "top1_greater_than_threshold": top1.is_greater_than(individual_threshold),
        "top1_percentage": top1.fill_percentage,
        "top2_percentage": top2.fill_percentage,
    }

    return results, sorted_checkboxes

def get_checkbox(b64_images=None, image_paths=None) -> list:
    """
    Process multiple images to detect selected checkboxes.

    Args:
        b64_images (list of str, optional): List of Base64-encoded image strings.
        image_paths (list of str, optional): List of paths to input images.

    Returns:
        list: List of selected checkbox class IDs corresponding to each image.
              -1 => No checkbox was selected
              -2 => No conclusive result
    """
    if b64_images:
        if not isinstance(b64_images, list):
            raise TypeError("b64_images should be a list of Base64-encoded strings.")
        # Decode all images
        images = []
        for idx, b64 in enumerate(b64_images):
            try:
                img = decode_base64_image(b64)
                images.append(img)
            except ValueError as e:
                print(f"Image decoding failed for image {idx}: {e}")
                images.append(None)
    elif image_paths:
        if not isinstance(image_paths, list):
            raise TypeError("image_paths should be a list of image file paths.")
        # Read all images
        images = []
        for idx, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Could not read the image from path: {path}.")
                images.append(None)
            else:
                images.append(img)
    else:
        raise ValueError("Either b64_images or image_paths must be provided as a list.")

    # Initialize list for selected class IDs
    selected_class_ids = []

    # Iterate over each image
    for img_idx, image in enumerate(images):
        if image is None:
            print(f"Skipping image {img_idx} due to previous errors.")
            selected_class_ids.append(-2)  # No conclusive result
            continue

        try:
            # Process image to get sorted checkboxes
            sorted_checkboxes = process_image_data(image, img_size=512, threshold=127, margin_ratio=0.05)

            # Analyze top checkboxes
            if len(sorted_checkboxes) >= 2:
                results_analysis, sorted_checkboxes = analyze_top_checkboxes(
                    sorted_checkboxes, percentage_threshold=5, individual_threshold=4
                )
            elif len(sorted_checkboxes) == 1:
                # Handle single checkbox case
                top1 = sorted_checkboxes[0]
                results_analysis = {
                    "difference_greater_than_threshold": True,  # No second checkbox to compare
                    "top1_greater_than_threshold": top1.is_greater_than(4),
                    "top1_percentage": top1.fill_percentage,
                    "top2_percentage": 0.0,  # No second checkbox
                }
            else:
                raise ValueError("No valid checkboxes detected.")

            # Decision logic
            if results_analysis['difference_greater_than_threshold'] and results_analysis['top1_greater_than_threshold']:
                selected_class_id = sorted_checkboxes[0].class_id
            elif len(sorted_checkboxes) == 1 and results_analysis['top1_greater_than_threshold']:
                selected_class_id = sorted_checkboxes[0].class_id
            elif not results_analysis['top1_greater_than_threshold']:
                selected_class_id = -1  # No checkbox selected
            else:
                selected_class_id = -2  # Inconclusive

            if len(sorted_checkboxes) != EXPECTED_NUMBER_OF_CHECKBOXES:
                print(f"Detected {len(sorted_checkboxes)}")
                selected_class_id = -3

        except ValueError as e:
            print(f"Processing Error in image {img_idx}: {e}")
            selected_class_id = -2  # Inconclusive

        selected_class_ids.append(selected_class_id)

    return selected_class_ids

if __name__ == "__main__":
    t_0 = time.time()

    # TEST CASE
    path = "train_section_data2.json"

    with open(path, 'r') as f:
        content = json.load(f)

    # Extract Base64-encoded images
    b64_images = [item['question_selector'] for item in content]
    
    # Process images to get selected checkboxes
    selected_checkboxes = get_checkbox(b64_images=b64_images)

    t_1 = time.time()

    # Display results
    for idx, selected in enumerate(selected_checkboxes):
        if selected > 0:
            print(f"Selected Checkbox in Image {idx}: Number {selected}")
        elif selected == -1:
            print(f"Selected Checkbox in Image {idx}: None")
        elif selected == -2:
            print(f"Selected Checkbox in Image {idx}: Inconclusive")
        elif selected == -3:
            print(f"Selected Checkbox in Image {idx}: Wrong number of checkboxes detected")

    print(f"Analyzed {len(selected_checkboxes)} images in {t_1 - t_0:.2f} seconds.")
    
    # Collect results
    results = []
    for idx, selected in enumerate(selected_checkboxes):
        result = {
            "image_index": idx,
            "status": None,
            "selected_checkbox": None,
        }

        if selected > 0:
            result["status"] = "Selected"
            result["selected_checkbox"] = selected
        elif selected == -1:
            result["status"] = "None"
        elif selected == -2:
            result["status"] = "Inconclusive"
        elif selected == -3:
            result["status"] = "Wrong number of checkboxes detected"

        results.append(result)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
