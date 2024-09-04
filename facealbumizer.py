import os
import cv2
import numpy as np
from shutil import copyfile
import time

# Path to the directory where the captured photo will be saved
photo_directory = 'place the directory of the collection of photos'
output_directory = 'place the directory of the photos where the photos matching the face will be stored'

# Ensure photo and output directories exist
if not os.path.exists(photo_directory):
    os.makedirs(photo_directory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_photo(photo_directory):
    """Capture a photo from the webcam and save it to the photo directory."""
    cap = cv2.VideoCapture(0)
    
    print("Webcam will take a photo in 5 seconds. Get ready!")
    time.sleep(5)  # Wait for 5 seconds before capturing the image
    
    ret, frame = cap.read()
    if ret:
        photo_path = os.path.join(photo_directory, 'captured_image.jpg')
        cv2.imwrite(photo_path, frame)
        print(f"Photo captured and saved at {photo_path}")
    else:
        print("Failed to capture image from webcam.")
    
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(image_path):
    """Detect faces in an image and return the face regions."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def save_detected_face(image_path, faces):
    """Save cropped faces to the output directory."""
    image = cv2.imread(image_path)
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)

    for i, (x, y, w, h) in enumerate(faces):
        face_img = image[y:y+h, x:x+w]
        output_folder = os.path.join(output_directory, f"person_{i+1}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, f"{file_name}_face{i+1}{ext}")
        cv2.imwrite(output_path, face_img)

def process_photos(photo_directory):
    """Process all photos in the directory to detect and save faces."""
    for file_name in os.listdir(photo_directory):
        file_path = os.path.join(photo_directory, file_name)
        if os.path.isfile(file_path):
            faces = detect_faces(file_path)
            if len(faces) > 0:
                save_detected_face(file_path, faces)
            else:
                print(f"No faces detected in {file_name}")

if __name__ == "__main__":
    # Capture photo from webcam
    capture_photo(photo_directory)
    
    # Process the captured photo
    process_photos(photo_directory)
    
    print("Face detection and sorting complete.")
