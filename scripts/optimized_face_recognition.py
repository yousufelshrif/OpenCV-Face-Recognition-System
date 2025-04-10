import os
import cv2
import numpy as np
import time

class OptimizedFaceRecognition:
    def __init__(self):
        # Initialize face detector
        self.net = self._init_face_detector()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.confidence_threshold = 0.7
        self.frame_skip = 2  # Process every 3rd frame
        self.frame_count = 0
        
    def _init_face_detector(self):
        """Initialize OpenCV's DNN face detector"""
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        config_path = "deploy.prototxt"
        
        #Check for model files
        if not os.path.exists(model_path):
            print("Downloading face detection model files...")
            try:
                import urllib.request
                # Download the correct files
                print("Downloading model file...")
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    model_path
                )
                print("Downloading config file...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
                    config_path
                )
                print("Model files downloaded successfully!")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("\nAlternative solution:")
                print("1. Download these files manually:")
                print("   - https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/res10_300x300_ssd_iter_140000.caffemodel")
                print("   - https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt")
                print("2. Place both files in this directory:", os.getcwd())
                print("3. Run the script again")
                raise
        
        # Verify file sizes to ensure they were downloaded correctly
        if os.path.getsize(model_path) < 1000 or os.path.getsize(config_path) < 1000:
            print("Error: Downloaded files appear to be corrupted or too small")
            print("Please download the files manually from the URLs above")
            raise Exception("Model files are corrupted or incomplete")
        
        try:
            print("Loading face detection model...")
            # Load the Caffe model with correct parameters
            net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            print("Face detection model loaded successfully!")
            return net
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            print("The model files might be corrupted. Please try downloading them manually.")
            raise
    
    def detect_face(self, frame):
        """Detect faces in the frame using OpenCV DNN"""
        height, width = frame.shape[:2]
        scale = 0.5  # Reduce size for faster processing
        
        # Resize frame
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), 
                                    [104, 117, 123], False, False)
        self.net.setInput(blob)
        
        # Run detection
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get coordinates and scale back to original size
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                padding = int((x2 - x1)*10)  # 10% padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
                
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def preprocess_face(self, face_img):
        """Preprocess face image for recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (160, 160))
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def train(self, data_dir):
        """Train the face recognition model"""
        faces = []
        labels = []
        label_id = 0
        label_map = {}
        
        for user in os.listdir(data_dir):
            print(f"Processing user: {user}")
            user_dir = os.path.join(data_dir, user)
            if not os.path.isdir(user_dir):
                continue
                
            label_map[label_id] = user
            for img_name in os.listdir(user_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(user_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces.append(gray)
                        labels.append(label_id)
            
            label_id += 1
        
        self.face_recognizer.train(faces, np.array(labels))
        self.label_map = label_map
        self.face_recognizer.save('face_recognizer.yml')
        
    def recognize_face(self, frame):
        """Recognize faces in the frame"""
        self.frame_count += 1
        if self.frame_count % (self.frame_skip + 1) != 0:
            return frame, []
            
        faces = self.detect_face(frame)
        results = []
        
        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
                
            # Preprocess face
            gray = self.preprocess_face(face_img)
            
            # Recognize face
            try:
                label_id, confidence = self.face_recognizer.predict(gray)
                if confidence < 100:  # Lower confidence is better
                    name = self.label_map.get(label_id, "Unknown")
                    results.append((x1, y1, x2, y2, name, confidence))
            except:
                continue
                
        return frame, results
    
    def draw_results(self, frame, results):
        """Draw recognition results on the frame"""
        for (x1, y1, x2, y2, name, confidence) in results:
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw name and confidence
            text = f"{name} ({confidence:.1f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize face recognition system
    face_rec = OptimizedFaceRecognition()
    
    # Train the model (if not already trained)
    if not os.path.exists('face_recognizer.yml'):
        print("Training face recognition model...")
        face_rec.train('data')
    
    # Load trained model
    face_rec.face_recognizer.read('face_recognizer.yml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame, results = face_rec.recognize_face(frame)
        
        # Draw results
        frame = face_rec.draw_results(frame, results)
        
        # Show frame
        cv2.imshow('Face Recognition', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 