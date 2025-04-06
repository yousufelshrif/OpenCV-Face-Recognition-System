import os
import cv2
import numpy as np
import urllib.request

class FaceRecognition:
    def __init__(self):
        self.face_detector = self._init_face_detector()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        
    def _init_face_detector(self):
        """Initialize OpenCV's DNN face detector"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join("models", "res10")
        os.makedirs(models_dir, exist_ok=True)
        
        # Use os.path.join for cross-platform compatibility
        model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        config_path = os.path.join(models_dir, "deploy.prototxt")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Downloading face detection model files...")
            try:
                print("Downloading model file...")
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/res10_300x300_ssd_iter_140000.caffemodel",
                    model_path
                )
                print("Downloading config file...")
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt",
                    config_path
                )
                print("Model files downloaded successfully!")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("\nAlternative solution:")
                print("1. Create a directory structure: models/res10/")
                print("2. Download these files manually:")
                print("   - https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/res10_300x300_ssd_iter_140000.caffemodel")
                print("   - https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt")
                print("3. Place them in the models/res10/ directory")
                raise
        
        try:
            return cv2.dnn.readNetFromCaffe(config_path, model_path)
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            print("Please check if the model files are in the correct location and format")
            raise
    
    def detect_face(self, frame):
        """Detect faces in the frame"""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                    [104, 117, 123], False, False)
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                faces.append((x1, y1, x2, y2, confidence))
        
        return faces
    
    def collect_dataset(self):
        """Collect face dataset for a user"""
        user_id = input("Enter User ID or Name: ").strip()
        if user_id == "":
            raise Exception("The ID is empty")
            
        user_dir = os.path.join("data", user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        img_id = 0
        print(f"Starting data collection for user {user_id}...")
        print("Press 'Enter' to finish collection")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.detect_face(frame)
            if faces:
                x1, y1, x2, y2, confidence = faces[0]  # Use first detected face
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    img_id += 1
                    try:
                        face = cv2.resize(face_img, (200, 200))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        
                        file_name_path = os.path.join(user_dir, f"user.{user_id}.{img_id}.jpg")
                        cv2.imwrite(file_name_path, face)
                        
                        # Show image count and confidence
                        cv2.putText(frame, f"Images collected: {img_id}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        continue
            
            cv2.imshow("Face Collection", frame)
            
            if cv2.waitKey(1) == 13:  # Press "Enter" to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection for {user_id} completed. {img_id} images saved.")
    
    def train_model(self):
        """Train the face recognition model"""
        data_dir = "data"
        faces = []
        labels = []
        label_id = 0
        
        for user in os.listdir(data_dir):
            user_dir = os.path.join(data_dir, user)
            if not os.path.isdir(user_dir):
                continue
                
            self.label_map[label_id] = user
            print(f"Processing user: {user}")
            
            for img_name in os.listdir(user_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(user_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(label_id)
            
            label_id += 1
        
        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save("classifier.xml")
        print(f"Training completed. Model saved to classifier.xml")
    
    def recognize_faces(self):
        """Run face recognition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting face recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.detect_face(frame)
            for (x1, y1, x2, y2, confidence) in faces:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # Preprocess face
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))
                
                # Recognize face
                try:
                    id, pred = self.face_recognizer.predict(gray)
                    conf = int(100 * (1 - pred / 300))

                    if conf > 99:
                        name = self.label_map.get(id, "Unknown")
                        text = f"{name} (Conf: {conf:.1f})"
                    else:
                        text = "Unknown"
                except:
                    text = "Unknown"
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw detection confidence
                cv2.putText(frame, f"Detect: {confidence:.2f}", (x1, y1 - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw recognition result
                cv2.putText(frame, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    fr = FaceRecognition()
    
    while True:
        print("\n1. Collect Dataset")
        print("2. Train Model")
        print("3. Run Recognition")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            fr.collect_dataset()
        elif choice == "2":
            fr.train_model()
        elif choice == "3":
            fr.recognize_faces()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 