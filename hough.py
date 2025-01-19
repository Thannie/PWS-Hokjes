import os
import shutil
import random

def distribute_data(source_dir, train_dir, val_dir, train_ratio=0.75):
    """
    Distributes images and labels into training and validation folders.

    Args:
        source_dir (str): Path to the folder containing all images and labels.
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        train_ratio (float): Ratio of data to allocate to training (default is 0.75).
    """
    # Create the necessary directories if they don't exist
    for dir_path in [train_dir, val_dir]:
        os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)

    # Get lists of images and labels
    all_files = os.listdir(source_dir)
    images = [f for f in all_files if f.endswith(".png")]
    labels = [f for f in all_files if f.endswith(".txt")]

    # Identify matched pairs (image and label)
    matched_pairs = []
    for image in images:
        label = os.path.splitext(image)[0] + ".txt"
        if label in labels:
            matched_pairs.append((image, label))

    # Shuffle the matched pairs
    random.shuffle(matched_pairs)

    # Split the pairs into training and validation sets
    split_index = int(len(matched_pairs) * train_ratio)
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]

    # Copy files to the respective folders
    for pair, target_dir in [(train_pairs, train_dir), (val_pairs, val_dir)]:
        for image, label in pair:
            # Copy image
            shutil.copy(os.path.join(source_dir, image), os.path.join(target_dir, "images", image))
            # Copy label
            shutil.copy(os.path.join(source_dir, label), os.path.join(target_dir, "labels", label))

    print(f"Data distribution complete:")
    print(f"  Training set: {len(train_pairs)} pairs")
    print(f"  Validation set: {len(val_pairs)} pairs")

# Example usage
source_directory = r"unannotated_images\train"
train_output = r"dataset\train"
val_output = r"dataset\train"
distribute_data(source_directory, train_output, val_output)
