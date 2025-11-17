import os
import random

import numpy as np
from PIL import Image
from typing import Tuple, Dict, List

from src.config_loader import load_config

config = load_config()
random.seed(config['seed'])
np.random.seed(config['seed'])

def preprocess_image(image_path: str,
                     target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image as a flattened numpy array
    """
    try:
        img = Image.open(image_path)

        # convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # resize to target size, convert to float32 and normalize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0

        #return flattened array
        return img_array.flatten()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def select_random_subjects(total_subjects: int = 68,
                           n_subjects: int = 25) -> List[int]:
    """
    Randomly select subjects from the PIE dataset.

    Args:
        total_subjects: Total number of subjects in PIE (68)
        n_subjects: Number of subjects to select (25)

    Returns:
        List of selected subject IDs
    """
    all_subjects = list(range(1, total_subjects + 1))
    return sorted(random.sample(all_subjects, n_subjects))

def create_label_mapping(selected_subjects: List[int],
                         selfie_label: int = 25) -> Dict[int, int]:
    """
    Create a mapping from original subject IDs to consecutive class labels.

    Args:
        selected_subjects: List of selected subject IDs
        selfie_label: Label for selfie class (should be last)

    Returns:
        Dictionary mapping original subject IDs to new labels
    """
    label_mapping = {subject_id: i for i, subject_id in enumerate(selected_subjects)}
    label_mapping[selfie_label] = len(selected_subjects)
    return label_mapping

def load_cmupie_dataset(data_dir: str,
                        selected_subjects: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load PIE dataset for selected subjects.

    Args:
        data_dir: Directory containing the PIE dataset
        selected_subjects: List of selected subject IDs

    Returns:
        Tuple of (images, labels, subject_list)
    """
    images, labels, valid_subjects = [], [], []
    label_mapping = create_label_mapping(selected_subjects)
    valid_extensions = ['.png', '.jpg', '.jpeg']

    for subject_id in selected_subjects:
        subject_dir = os.path.join(data_dir, str(subject_id))

        if not os.path.exists(subject_dir):
            print(f"Warning: Directory for subject {subject_id} not found: {subject_dir}")
            continue

        valid_subjects.append(subject_id)

        # load all images for this subject
        for image_file in os.listdir(subject_dir):
            if any(image_file.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(subject_dir, image_file)
                processed_image = preprocess_image(image_path)

                if processed_image is not None:
                    images.append(processed_image)
                    labels.append(label_mapping[subject_id])

    return np.array(images), np.array(labels), valid_subjects

def load_selfie_images(selfie_dir: str,
                       n_train: int = 7,
                       n_test: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load selfie images and split into train and test sets.

    Args:
        selfie_dir: Directory containing selfie images
        n_train: Number of selfie images for training
        n_test: Number of selfie images for testing

    Returns:
        Tuple of (train_images, test_images, train_labels, test_labels)
    """
    SELFIE_LABEL = 400  # selfie class label
    selfie_images = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

    # load all selfie images
    for image_file in sorted(os.listdir(selfie_dir)):
        if any(image_file.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(selfie_dir, image_file)
            processed_image = preprocess_image(image_path)

            if processed_image is not None:
                selfie_images.append(processed_image)

    if len(selfie_images) < n_train + n_test:
        raise ValueError(f"Not enough selfie images. Found {len(selfie_images)}, need {n_train + n_test}")

    # convert to numpy array and split
    selfie_images = np.array(selfie_images)
    train_images = selfie_images[:n_train]
    test_images = selfie_images[n_train:n_train + n_test]

    # create labels
    train_labels = np.full(n_train, SELFIE_LABEL)
    test_labels = np.full(n_test, SELFIE_LABEL)

    return train_images, test_images, train_labels, test_labels

def split_dataset(images: np.ndarray, labels: np.ndarray, test_size: float = 0.3, seed: int = config.get('seed', 963)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.

    Args:
        images: Image data
        labels: Corresponding labels
        test_size: Proportion of data for testing (0.3 = 30%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x_train, x_test, y_train, y_test)
    """
    n_samples = len(images)
    n_test = int(n_samples * test_size)

    # generate random permutation of indices
    indices = np.random.permutation(n_samples)
    train_indices, test_indices = indices[n_test:], indices[:n_test]

    x_train, x_test = images[train_indices], images[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    return x_train, x_test, y_train, y_test

def prepare_dataset(cmupie_dir: str, selfie_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Prepare the complete dataset for Part 1.

    Args:
        cmupie_dir: Directory containing PIE dataset
        selfie_dir: Directory containing selfie images

    Returns:
        Tuple of (x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices)
    """
    #select random subjects
    selected_subjects = select_random_subjects()
    print(f"Selected subjects: {selected_subjects}")

    #load and split the dataset - PIE and selfies
    images, labels, valid_subjects = load_cmupie_dataset(cmupie_dir, selected_subjects)
    x_train_cmupie, x_test_cmupie, y_train_cmupie, y_test_cmupie = split_dataset(images, labels)
    x_train_selfie, x_test_selfie, y_train_selfie, y_test_selfie = load_selfie_images(selfie_dir)

    print(f"Selected {len(x_train_selfie)} subjects from the selfie dataset for training.")

    #combine data - vertical stacking for images, horizontal for labels
    x_train = np.vstack([x_train_cmupie, x_train_selfie])
    x_test = np.vstack([x_test_cmupie, x_test_selfie])
    y_train = np.hstack([y_train_cmupie, y_train_selfie])
    y_test = np.hstack([y_test_cmupie, y_test_selfie])

    #keep track of selfie indices for highlighting
    train_selfie_indices = np.arange(len(x_train_cmupie), len(x_train))
    test_selfie_indices = np.arange(len(x_test_cmupie), len(x_test))

    return x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices