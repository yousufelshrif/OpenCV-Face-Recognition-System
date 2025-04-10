import cv2
import os

def init_face_detector():
    """Initialize OpenCV's DNN face detector"""
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "deploy.prototxt"
    
    if not os.path.exists(model_path):
        print("Downloading face detection model files...")
        try:
            import urllib.request
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
            print("\nPlease download these files manually:")
            print("1. https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/res10_300x300_ssd_iter_140000.caffemodel")
            print("2. https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt")
            raise
    
    return cv2.dnn.readNetFromCaffe(config_path, model_path)

def detect_faces(frame, net):
    """Detect faces in the frame"""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            faces.append((x1, y1, x2, y2, confidence))
    
    return faces

def main():
    # Initialize face detector
    print("Initializing face detector...")
    net = init_face_detector()
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting face detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect faces
        faces = detect_faces(frame, net)
        
        # Draw detection results
        for (x1, y1, x2, y2, confidence) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Face Detection Test', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed")

if __name__ == "__main__":
    main()
