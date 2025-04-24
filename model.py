import cv2
import os
import numpy as np

emotions = ["angry", "happy", "sad", "neutral"]
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fisher_face = cv2.face.FisherFaceRecognizer_create()

def create_dirs():
    for emotion in emotions:
        path = f"dataset/{emotion}"
        if not os.path.exists(path):
            os.makedirs(path)

def capture_faces(emotion):
    print(f"\nShow emotion: {emotion.upper()} — hold still for a few seconds...")
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 16:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (350, 350))
            file_path = f"dataset/{emotion}/{count}.jpg"
            cv2.imwrite(file_path, face_img)
            count += 1
            print(f"Captured: {file_path}")
        
        cv2.imshow("Capturing...", frame)
        if cv2.waitKey(1) == 27:  # Esc to break
            break

    cap.release()
    cv2.destroyAllWindows()

def prepare_data():
    images = []
    labels = []
    label_map = {emotion: idx for idx, emotion in enumerate(emotions)}

    for emotion in emotions:
        files = os.listdir(f"dataset/{emotion}")
        for file in files:
            img_path = f"dataset/{emotion}/{file}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label_map[emotion])

    return images, np.array(labels)

def train_model():
    images, labels = prepare_data()
    fisher_face.train(images, labels)
    fisher_face.save("model.xml")
    print("✅ Model trained and saved as model.xml")

# Run
create_dirs()
for emotion in emotions:
    capture_faces(emotion)
train_model()
