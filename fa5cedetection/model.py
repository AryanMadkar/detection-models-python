import cv2
import time
import mediapipe as mp

class FaceDetector:
    def __init__(self, detection_confidence=0.75):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=detection_confidence)
    
    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        face_data = []
        increase_value = 1
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                bbox = int(bboxC.xmin * w * increase_value), int(bboxC.ymin * h * increase_value), int(bboxC.width * w * increase_value), int(bboxC.height * h * increase_value)
                face_data.append(bbox)
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
        
        return img, face_data

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    face_detector = FaceDetector()
    pTime = time.time()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: Unable to read from camera.")
            break
        
       
        img, face_data = face_detector.find_faces(img)
        print("Face detected" , face_data)
        
        # FPS Calculation
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        cv2.putText(img, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand and Face Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
