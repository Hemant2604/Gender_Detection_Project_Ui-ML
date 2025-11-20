from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image path - update this to your image path or use command line argument
image_path = input("Enter image path (or press Enter for default): ").strip()
if not image_path:
    # Default image path - update this to your actual image path
    image_path = os.path.join(BASE_DIR, "test_image.jpg")

image = cv2.imread(image_path)

if image is None:
    print(f"Could not read input image from: {image_path}")
    print("Please make sure the image path is correct.")
    exit()

# Load pre-trained model - using os.path.join for cross-platform compatibility
model_path = os.path.join(BASE_DIR, 'GenderDetector', 'models', "epochs_044-val_accuracy_0.966.keras") 

try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Model path: {model_path}")
    exit()

# Detect faces in the image
faces, face_confidences = cv.detect_face(image)

if len(faces) == 0:
    print("No faces detected in the image.")
    exit()

classes = ['man', 'woman']

# Loop through detected faces
for face_idx, face_box in enumerate(faces):
    # Get corner points of face rectangle
    (startX, startY) = face_box[0], face_box[1]
    (endX, endY) = face_box[2], face_box[3]

    # Draw rectangle over face
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Crop the detected face region
    face_crop = np.copy(image[startY:endY, startX:endX])

    # Preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # Apply gender detection on face
    conf = model.predict(face_crop, verbose=0)[0]
    print(f"Face {face_idx + 1} - Confidence scores: {conf}")

    # Get label with max accuracy
    gender_idx = np.argmax(conf)
    label = classes[gender_idx]
    confidence_score = conf[gender_idx] * 100

    label_text = "{}: {:.2f}%".format(label, confidence_score)
    print(f"Face {face_idx + 1} - Detected: {label_text}")

    # Position label above face rectangle
    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # Write label and confidence above face rectangle
    if confidence_score > 50.0:
        cv2.putText(image, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

# Save output
output_path = os.path.join(BASE_DIR, "gender_detection.jpg")
cv2.imwrite(output_path, image)
print(f"\nResult saved to: {output_path}")

# Optional: Display output (uncomment if you want to see the image)
# cv2.imshow("gender detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()