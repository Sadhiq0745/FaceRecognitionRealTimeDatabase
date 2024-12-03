import numpy as np
import cv2
import os
import json

from keras.src.saving import load_model


def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to an image.
    :param image: Input image (BGR format).
    :param gamma: Gamma value for correction. Default is 1.0 (no change).
    :return: Gamma-corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_face(image_path, gamma=1.0, target_size=(160, 160)):
    """
    Preprocess the face image for FaceNet with gamma correction.
    :param image_path: Path to the input image.
    :param gamma: Gamma value for correction.
    :param target_size: Target size for resizing the image.
    :return: Preprocessed image array.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    image_gamma_corrected = gamma_correction(image, gamma)
    image_rgb = cv2.cvtColor(image_gamma_corrected, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)


def compute_face_embeddings(image_dir, model_path, output_path, gamma=1.0):
    """
    Compute FaceNet embeddings for all images in a directory with gamma correction.
    :param image_dir: Directory containing input images.
    :param model_path: Path to the FaceNet model.
    :param output_path: Path to save the embeddings.
    :param gamma: Gamma value for correction.
    """
    # Load the FaceNet model
    model = load_model(model_path)

    embeddings = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {image_path}...")

            processed_face = preprocess_face(image_path, gamma)
            if processed_face is not None:
                embedding = model.predict(processed_face)
                embeddings[filename] = embedding.tolist()  # Convert to list for JSON serialization

    # Save embeddings to a JSON file
    with open(output_path, 'w') as f:
        json.dump(embeddings, f, indent=4)
    print(f"Embeddings saved to {output_path}")


# Example usage
if __name__ == "__main__":
    input_directory = "input_faces"  # Replace with your input faces folder
    model_file = "facenet_keras.h5"  # Replace with the path to your FaceNet model
    output_file = "embeddings.json"  # Replace with your desired output JSON file
    gamma_value = 1.2  # Adjust gamma as needed

    compute_face_embeddings(input_directory, model_file, output_file, gamma_value)
