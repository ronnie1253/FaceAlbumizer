import cv2
import os


def capture_face_image(output_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    timeout = 100  # 100 frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1
        cv2.imshow('Press "s" to save and exit', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(output_path, frame)
            break
        elif frame_count > timeout:
            print("Timeout: Could not capture image within the time frame.")
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def extract_face_roi(image, faces):
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_roi = image[y:y + h, x:x + w]
    return face_roi


def search_for_face_in_folder(reference_face_roi, folder_path, output_folder, face_cascade):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            faces = detect_faces(img, face_cascade)

            for (x, y, w, h) in faces:
                face_roi = img[y:y + h, x:x + w]
                similarity = cv2.matchTemplate(face_roi, reference_face_roi, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(similarity)
                if max_val >= 0.6:
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, img)
                    print(f"Matching face detected and saved: {output_path}")


reference_image_path = r"C:\Users\syunm\OneDrive\Pictures\ref_image"
folder_path = r"C:\Users\syunm\OneDrive\Pictures\test_a"
output_folder = r"C:\Users\syunm\OneDrive\Pictures\output_b"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture_face_image(reference_image_path)

reference_image = cv2.imread(reference_image_path)
reference_faces = detect_faces(reference_image, face_cascade)

if len(reference_faces) == 0:
    print("No face detected in the reference image.")
else:
    reference_face_roi = extract_face_roi(reference_image, reference_faces)
    search_for_face_in_folder(reference_face_roi, folder_path, output_folder, face_cascade)
