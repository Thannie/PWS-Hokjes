import torch
import cv2
import base64
import json
import numpy as np
import time
import os
import warnings

# Suppress FutureWarnings temporarily
warnings.filterwarnings("ignore", category=FutureWarning)

EXPECTED_NUMBER_OF_CHECKBOXES: int

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
    
def visualize_detections(image, checkbox_data, output_path, prefix="before", selected_index=None):
    """
    Draw bounding boxes on a copy of 'image' to visualize detections, highlighting the selected checkbox.

    Args:
        image (numpy.ndarray): The original image (BGR).
        checkbox_data (list): List of tuples: (Checkbox, y1, x1, x2, y2).
        output_path (str): Folder path where you want to save the image.
        prefix (str): Prefix to add to saved file name (e.g., "before" or "after").
        selected_index (int, optional): The index of the selected checkbox to highlight.
    """
    # Make a copy so we don't mutate the original image
    img_copy = image.copy()

    for idx, (checkbox, y1, x1, x2, y2) in enumerate(checkbox_data):
        # Default color is green
        color = (0, 255, 0)  # Green

        # If the current checkbox is the selected one, use red
        if idx == selected_index:
            color = (0, 0, 255)  # Red

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Add text label with index and fill percentage
        cv2.putText(
            img_copy,
            f"#{idx+1} {checkbox.fill_percentage:.1f}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Construct filename and save
    filename = f"{prefix}_detections.png"
    save_path = os.path.join(output_path, filename)
    cv2.imwrite(save_path, img_copy)
    print(f"Saved {prefix} visualization: {save_path}")

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

def rank_and_filter_checkboxes(
    checkbox_data,
    expected_count,
    overlap_threshold=0.5,  # Step 1 overlap IoU threshold
    size_tolerance=0.2,     # For final ranking
    horizontal_tolerance=10  # For final ranking
):
    """
    1) Prune checkboxes that overlap more than `overlap_threshold`.
    2) If still more than `expected_count`, prune by x-position (closest to mean x-center).
    3) Rank remaining checkboxes by prior 'likelihood' logic (size/horizontal alignment),
       return top `expected_count` boxes.
    """

    if not checkbox_data:
        return []

    def compute_iou(boxA, boxB):
        """
        Intersection-over-Union for bounding boxes.
        box = (checkbox, y1, x1, x2, y2).
        """
        _, Ay1, Ax1, Ax2, Ay2 = boxA
        _, By1, Bx1, Bx2, By2 = boxB

        inter_x1 = max(Ax1, Bx1)
        inter_y1 = max(Ay1, By1)
        inter_x2 = min(Ax2, Bx2)
        inter_y2 = min(Ay2, By2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
        areaB = (Bx2 - Bx1) * (By2 - By1)

        union_area = areaA + areaB - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    # Step 1: Prune by overlap
    # Sort to ensure consistent ordering (e.g., top-left to bottom-right)
    checkbox_data = sorted(checkbox_data, key=lambda x: (x[1], x[2]))
    pruned = []
    for box in checkbox_data:
        should_keep = True
        for kept in pruned:
            iou = compute_iou(box, kept)
            if iou > overlap_threshold:
                should_keep = False
                break
        if should_keep:
            pruned.append(box)

    # If we've hit the exact count after overlap pruning, return them (unranked).
    if len(pruned) == expected_count:
        return [cb[0] for cb in pruned]

    # Step 2: If still too many, prune by x-position alignment (keep those closest to mean x-center)
    if len(pruned) > expected_count:
        mean_x_center = np.mean([(x1 + x2) / 2 for _, y1, x1, x2, y2 in pruned])
        pruned.sort(key=lambda b: abs(((b[2] + b[3]) / 2) - mean_x_center))
        pruned = pruned[:expected_count]

    # If we've hit the exact count, we can return here
    if len(pruned) == expected_count:
        return [cb[0] for cb in pruned]

    # Step 3: Rank remaining boxes by your previous "likelihood" (size & horizontal alignment)
    centers_x = [(x1 + x2) / 2 for _, y1, x1, x2, y2 in pruned]
    heights = [(y2 - y1) for _, y1, x1, x2, y2 in pruned]
    widths = [(x2 - x1) for _, y1, x1, x2, y2 in pruned]

    mean_x_center = np.mean(centers_x)
    mean_h = np.mean(heights)
    mean_w = np.mean(widths)

    ranked = []
    for (chk, y1, x1, x2, y2) in pruned:
        cx = (x1 + x2) / 2
        w = x2 - x1
        h = y2 - y1

        # Horizontal alignment
        score_h = 1.0 if abs(cx - mean_x_center) <= horizontal_tolerance else 0.0

        # Size alignment
        w_ok = abs(w - mean_w) / mean_w <= size_tolerance
        h_ok = abs(h - mean_h) / mean_h <= size_tolerance
        score_size = 1.0 if (w_ok and h_ok) else 0.0

        total_score = score_h + score_size
        ranked.append((chk, total_score))

    # Sort by score DESC
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Take top `expected_count`
    final_selection = [checkbox for checkbox, _ in ranked[:expected_count]]
    return final_selection



def analyze_fill_level(checkbox_img, threshold=127, fixed_margin=10):
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


def process_image_data(image, img_size=512, threshold=127, margin_ratio=0.05, visualize=False, output_path=".", idx=""):
    """
    Processes an image to detect and filter checkboxes, and visualize the results.
    """
    # Perform inference
    detections = get_detections(model, image, img_size)

    # Build checkbox data
    checkbox_data = []
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop and analyze fill level
        checkbox_img = image[y1:y2, x1:x2]
        try:
            fill_percentage = analyze_fill_level(checkbox_img, threshold)
        except ValueError:
            continue

        cbox = Checkbox(class_id=None, fill_percentage=fill_percentage)
        checkbox_data.append((cbox, y1, x1, x2, y2))

    # Optional: visualize detections before filtering
    if visualize:
        visualize_detections(
            image,
            checkbox_data,
            output_path=output_path,
            prefix=f"{idx}-before"
        )

    # Filter checkboxes
    filtered_checkboxes = rank_and_filter_checkboxes(
        checkbox_data,
        expected_count=EXPECTED_NUMBER_OF_CHECKBOXES
    )

    # Sort filtered checkboxes by fill_percentage in descending order
    sorted_checkboxes = sorted(filtered_checkboxes, reverse=True, key=lambda cb: cb.fill_percentage)

    # Optional: visualize detections after filtering
    final_data = [(cb, y1, x1, x2, y2) for cb, y1, x1, x2, y2 in checkbox_data if cb in filtered_checkboxes]

    # Sort checkboxes by their y and x positions (top to bottom, left to right)
    final_data_sorted = sorted(final_data, key=lambda x: (x[1], x[2]))  # Sort by y1, then x1

    # Extract sorted checkboxes based on position
    sorted_checkboxes = [cb for cb, _, _, _, _ in final_data_sorted]

    # Assign class IDs based on positional order
    for idx, checkbox in enumerate(sorted_checkboxes, start=1):
        checkbox.class_id = idx

    # Determine the selected checkbox (top filled)
    selected_checkbox = sorted_checkboxes[0] if sorted_checkboxes else None
    selected_index = None
    if selected_checkbox:
        # Find the index of the selected_checkbox in final_data
        for i, (cb, y1, x1, x2, y2) in enumerate(final_data):
            if cb == selected_checkbox:
                selected_index = i
                break

    if visualize:
        visualize_detections(
            image,
            final_data,
            output_path=output_path,
            prefix=f"{idx}-after",
            selected_index=selected_index
        )

    # Assign IDs based on sorted_checkboxes
    for idx, checkbox in enumerate(sorted_checkboxes, start=1):
        checkbox.class_id = idx

    return sorted_checkboxes




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

def get_checkbox(num_of_checkboxes:int, b64_images=None, image_paths=None) -> list:
    """
    Process multiple images to detect selected checkboxes.

    Args:
        num_of_checkboxes (int): How many checkboxes should be detected.
        b64_images (list of str, optional): List of Base64-encoded image strings.
        image_paths (list of str, optional): List of paths to input images.

    Returns:
        list: List of selected checkbox class IDs corresponding to each image.
              -1 => No checkbox was selected
              -2 => No conclusive result
    """
    global EXPECTED_NUMBER_OF_CHECKBOXES
    EXPECTED_NUMBER_OF_CHECKBOXES = num_of_checkboxes

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
            sorted_checkboxes = process_image_data(image, img_size=512, threshold=127, margin_ratio=0.05, visualize=False, output_path="detected_checkboxes", idx=img_idx)

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

    return_list = []

    for idx, selected in enumerate(selected_class_ids):
        return_list.append(
            {
                'request_number': idx,
                'selected_checkbox': selected
            })

    return return_list

if __name__ == "__main__":
    t_0 = time.time()

    # TEST CASE
    path = "train_section_data2.json"
    b_64 = []

    with open(path, 'r') as f:
        data = json.load(f)

    for obj in data:
        b_64.append(obj['question_selector'])

    # Process images to get selected checkboxes
    selected_checkboxes = get_checkbox(b64_images=b_64)

    t_1 = time.time()


    print(f"Analyzed {len(selected_checkboxes)} images in {t_1 - t_0:.2f} seconds.")
    
    # Collect results
    results = []
    for idx, obj in enumerate(selected_checkboxes):
        result = {
            "image_index": idx,
            "status": None,
            "selected_checkbox": None,
        }

        selected = obj['selected_checkbox']


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
