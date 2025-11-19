import os
import cv2
import cvlib as cv
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from .models import ImageUpload
from keras.utils import img_to_array
from keras.models import load_model
from PIL import Image
import uuid

# Load model once globally for performance
BASE_DIR = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, 'GenderDetector', 'model', 'epochs_044-val_accuracy_0.966.keras')
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

classes = ['man', 'woman']


def home(request):
    """Home page with image upload form"""
    return render(request, 'detect/upload.html')


def detect_gender(request):
    """Handle image upload and perform gender detection"""
    if request.method == 'POST' and request.FILES.get('image'):
        # Save uploaded image
        uploaded_image = request.FILES['image']
        image_upload = ImageUpload.objects.create(image=uploaded_image)
        
        # Get the full path to the uploaded image
        image_path = image_upload.image.path
        
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            return render(request, 'detect/result.html', {
                'error': 'Could not read the uploaded image. Please try another image.',
                'image_upload': image_upload
            })
        
        # Detect faces in the image
        faces, confidences = cv.detect_face(image)
        
        if len(faces) == 0:
            return render(request, 'detect/result.html', {
                'error': 'No face found, please upload another image',
                'image_upload': image_upload
            })
        
        predictions = []

        # Process each detected face
        for idx, face in enumerate(faces):
            # Get corner points of face rectangle
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            
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
            if model is not None:
                conf = model.predict(face_crop, verbose=0)[0]
                
                # Get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]
                confidence = conf[idx] * 100
                
                # Format label text
                label_text = f"{label}: {confidence:.2f}%"
                
                # Position label above face rectangle
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                
                # Write label and confidence above face rectangle
                cv2.putText(image, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)

                predictions.append({
                    'face_number': idx + 1,
                    'label': label,
                    'confidence': confidence,
                    'label_text': label_text
                })
        
        # Save processed image
        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        # Save the processed image
        cv2.imwrite(result_path, image)
        
        # Save result image path to model
        image_upload.result_image.name = f'results/{result_filename}'
        image_upload.save()
        
        # Prepare context for result page
        context = {
            'image_upload': image_upload,
            'num_faces': len(faces),
            'predictions': predictions,
        }
        
        return render(request, 'detect/result.html', context)
    
    return redirect('home')

