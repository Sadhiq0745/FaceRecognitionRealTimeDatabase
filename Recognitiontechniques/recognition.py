from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to the input image."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)
def detect_faces(images, gamma=1.0):
    # Initialize the MTCNN detector
    detector = MTCNN()
                                                         # Read and correct the input image
    image = cv2.imread(images)
    image_gamma_corrected = gamma_correction(image, gamma)
    image_rgb = cv2.cvtColor(image_gamma_corrected, cv2.COLOR_BGR2RGB)
    # Detect faces
    detections = detector.detect_faces(image_rgb)
                                                 # Draw bounding boxes around detected faces
    for face in detections:
        x, y, width, height = face['box']
        cv2.rectangle(image_gamma_corrected, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                                                          # Show the result
    plt.imshow(cv2.cvtColor(image_gamma_corrected, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Faces Detected by MTCNN (Gamma Corrected)")
    plt.show()
                                                                            # Example usage
if __name__ == "__main__":
    # Replace 'path/to/image.jpg' with an actual image path
    detect_faces('Images.jpg', gamma=1.5)
