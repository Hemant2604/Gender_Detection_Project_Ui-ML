# Gender Detection Django Web Application

A complete Django web application for detecting faces and classifying gender using a pre-trained TensorFlow/Keras model.

## Features

- 🖼️ Image upload interface with modern Bootstrap UI
- 👤 Automatic face detection using OpenCV and cvlib
- 🎯 Gender classification (man/woman) with confidence scores
- 📊 Visual results with bounding boxes and labels
- ⚡ Optimized model loading (loaded once globally)
- 🛡️ Error handling for edge cases

## Project Structure

```
GenderDetector/
├── GenderDetector/          # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── detect/                  # Django app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   └── urls.py
├── templates/
│   └── detect/
│       ├── upload.html
│       └── result.html
├── static/
│   └── detect/
│       └── css/
│           └── style.css
├── media/
│   ├── uploads/            # User uploaded images
│   └── results/            # Processed result images
├── manage.py
├── requirements.txt
├── epochs_044-val_accuracy_0.966.keras  # Trained model
└── README.md
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Migrations

```bash
python manage.py migrate
```

### 3. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 4. Run Development Server

```bash
python manage.py runserver
```

### 5. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## Usage

1. **Upload Image**: Click on the upload form and select an image containing faces
2. **Detection**: The system will automatically:
   - Detect all faces in the image
   - Crop and preprocess each face (resize to 96x96, normalize)
   - Classify gender using the trained model
   - Draw bounding boxes and labels on the image
3. **View Results**: See both the original and processed images with gender predictions

## Model Details

- **Model File**: `epochs_044-val_accuracy_0.966.keras`
- **Input Size**: 96x96 pixels
- **Preprocessing**:
  - Resize to 96x96
  - Normalize pixel values (/255.0)
  - Convert to array
  - Expand dimensions
- **Classes**: ['man', 'woman']
- **Output**: Confidence scores for each class

## Error Handling

- If no face is detected: Shows message "No face found, please upload another image"
- If image cannot be read: Shows appropriate error message
- Model loading errors are handled gracefully

## Technologies Used

- **Django**: Web framework
- **TensorFlow/Keras**: Deep learning model
- **OpenCV**: Image processing
- **cvlib**: Face detection
- **NumPy**: Numerical operations
- **Pillow**: Image handling
- **Bootstrap 5**: UI framework

## Development Notes

- Model is loaded once globally in `views.py` for better performance
- Processed images are saved in `media/results/` folder
- Original images are stored in `media/uploads/`
- All images are served via Django's media URL configuration

## Production Deployment

Before deploying to production:

1. Set `DEBUG = False` in `settings.py`
2. Update `SECRET_KEY` with a secure random key
3. Configure `ALLOWED_HOSTS`
4. Set up proper static file serving
5. Use a production-grade database (PostgreSQL recommended)
6. Configure proper media file storage (AWS S3, etc.)

## License

This project is open source and available for educational purposes.

